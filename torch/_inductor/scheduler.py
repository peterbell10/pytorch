import collections
import dataclasses
import functools
import itertools
import logging
import os
import pprint
import textwrap
from typing import Dict, List, Optional, Set, Tuple, Union, Callable

import sympy

import torch
from torch._dynamo.utils import dynamo_timed

from . import config, dependencies, ir, metrics
from .dependencies import canonicalization_prefix, MemoryDep, StarDep, WeakDep
from .sizevars import SimplifyIndexing
from .utils import cache_on_self, cmp, has_triton, sympy_subs, sympy_symbol
from .virtualized import V

log = logging.getLogger(__name__)


def pformat(obj):
    if isinstance(obj, set):
        # pformat has trouble with sets of sympy exprs
        obj = sorted(obj, key=str)
    result = pprint.pformat(obj, indent=4)
    if "\n" in result:
        return f"\n{textwrap.indent(result, ' '*4)}"
    return result


class OutputNode:
    def __init__(self, dep):
        self.unmet_dependencies = {dep}
        self.inverse_users = []

    def is_reduction(self):
        return False

    def get_alias_names(self):
        return ()

    def get_name(self):
        return "OUTPUT"

    __repr__ = get_name


class BaseSchedulerNode:
    def __init__(self, scheduler: "Scheduler", node: ir.Buffer):
        self.scheduler: "Scheduler" = scheduler
        self.node: ir.Buffer = node
        self.users: Optional[List[NodeUser]] = None
        self.inverse_users: List[BaseSchedulerNode] = []
        self.set_read_writes(node.get_read_writes())
        self.recursive_predecessors: Optional[Set[str]] = None
        self.min_order: Optional[int] = None
        self.max_order: Optional[int] = None
        self.last_usage: Set[str] = None  # buffers that won't be used after this kernel
        self.written = False

    def __repr__(self):
        return f"{type(self).__name__}(name={self.get_name()!r})"

    def debug_str(self):
        """Longer form printout for trace logs"""
        name = self.get_name()
        lines = [
            f"{name}: {type(self).__name__}({type(self.node).__name__})",
            f"{name}.writes = {pformat(self.read_writes.writes)}",
            f"{name}.unmet_dependencies = {pformat(self.unmet_dependencies)}",
            f"{name}.met_dependencies = {pformat(self.read_writes.reads - self.unmet_dependencies)}",
        ]
        try:
            lines += [
                self.debug_str_extra(),
            ]
        except Exception:
            log.warning("Ignoring error in debug_str()", exc_info=True)
        return "\n".join(lines).rstrip()

    def debug_str_extra(self):
        return ""

    def log_details(self):
        log.info(
            "%s: unmet_dependencies = %s, writes = %s",
            self,
            self.unmet_dependencies,
            self.read_writes.writes,
        )

    def update_mutated_names(self, renames: Dict[str, str]):
        self.set_read_writes(self.read_writes.rename(renames))

    def add_mutation_dep(self, dep):
        self.set_read_writes(self.read_writes.with_read(dep))

    def set_users(self, users: List["NodeUser"]):
        # deduplicate
        result: Dict[int, NodeUser] = {}
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = NodeUser(
                    use.node, result[id(use.node)].can_inplace and use.can_inplace
                )
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def get_aliases(self):
        return self.node.get_alias_names()

    def get_mutations(self):
        return self.node.get_mutation_names()

    def has_aliasing_or_mutation(self):
        return bool(self.get_aliases() or self.get_mutations())

    def set_read_writes(self, rw: dependencies.ReadWrites):
        self.read_writes: dependencies.ReadWrites = rw
        self.unmet_dependencies = self.read_writes.reads
        self.prune_deps()

    def used_buffer_names(self) -> Set[str]:
        return {
            dep.name
            for dep in itertools.chain(self.read_writes.reads, self.read_writes.writes)
        }

    def prune_deps(self):
        self.unmet_dependencies = {
            dep
            for dep in self.unmet_dependencies
            if dep.name not in self.scheduler.available_buffer_names
        }

    def prune_redundant_deps(self, name_to_fused_node):
        """
        Prunes stardeps intended for mutation ordering
        on an upstream fused node if after fusion there is another dependency
        on the fused upstream node, making the stardep redundant

        In essence this enforces an ordering on fusions. As fusions occur, prunable stardeps will
        be incrementally removed, enabling other fusions, ensuring they are fused in order.
        """
        name_to_dep_count = collections.Counter()

        for dep in self.unmet_dependencies:
            if not isinstance(dep, WeakDep):
                name_to_dep_count[name_to_fused_node[dep.name].get_name()] += 1

        def should_prune(dep):
            if isinstance(dep, WeakDep):
                is_redundant = (
                    name_to_dep_count[name_to_fused_node[dep.name].get_name()] > 0
                )
                # These can occur because fused nodes always gather deps from their snodes
                # If B has a weakdep on A
                # B gets fused with C, then any time BC is fused, the weakdep will reappear
                is_self_dep = name_to_fused_node[dep.name] == self
                return is_redundant or is_self_dep
            else:
                return False

        deps_to_prune = {dep for dep in self.unmet_dependencies if should_prune(dep)}
        self.unmet_dependencies = self.unmet_dependencies - deps_to_prune
        self.set_read_writes(self.read_writes.remove_reads(deps_to_prune))

    def get_name(self) -> str:
        return self.node.get_name()

    def get_first_name(self) -> str:
        return self.get_name()

    def get_names(self) -> Set[str]:
        return {self.get_name()}

    def get_nodes(self) -> List["BaseSchedulerNode"]:
        return [self]

    def get_device(self):
        return self.node.get_device()

    def is_reduction(self):
        return False

    def is_template(self):
        return False

    def is_extern(self):
        return False

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        return False

    def allocate(self):
        if not self.node.should_allocate():
            return

        if isinstance(self, (SchedulerNode,)) and (
            self.node.get_alias_names() or self.node.get_mutation_names()
        ):
            V.graph.wrapper_code.codegen_allocation(self.node)
            return

        if (
            (
                isinstance(self, (SchedulerNode,))
                # o what have i done.  lets make this an api
                or (
                    isinstance(self, ExternKernelSchedulerNode)
                    and isinstance(self.node, ir.AllReduce)
                )
            )
            and config.inplace_buffers
            and (
                not isinstance(V.kernel, torch._inductor.codegen.triton.TritonKernel)
                or getattr(V.kernel, "mutations", None) is not None
            )
        ):
            from .codegen.wrapper import buffer_reuse_key

            ordered_reads = sorted(self.read_writes.reads, key=lambda x: x.name)

            for read in ordered_reads:
                input_node: BaseSchedulerNode = self.scheduler.name_to_node.get(
                    read.name
                )
                if input_node and V.graph.wrapper_code.can_reuse(input_node):
                    remaining_uses = [
                        x
                        for x in input_node.users
                        if x.node.get_name()
                        not in self.scheduler.available_buffer_names
                    ]
                    if (
                        len(remaining_uses) == 1
                        and remaining_uses[0].can_inplace
                        and remaining_uses[0].node is self
                        and not isinstance(
                            input_node.node.get_layout(),
                            (
                                ir.MultiOutputLayout,
                                ir.MutationLayout,
                                ir.AliasedLayout,
                            ),
                        )
                        and buffer_reuse_key(input_node.node)
                        == buffer_reuse_key(self.node)
                    ):
                        V.graph.wrapper_code.codegen_inplace_reuse(
                            input_node.node, self.node
                        )
                        # hacky check for if V.kernel is a real kernel or NullHandler
                        if hasattr(V.kernel, "args"):
                            # if there isn't a triton kernel, then we don't need to call triton-specific things.
                            # but TODO this might be a convenient place to signal to the Collective kernels to inplace
                            # (and, can we make "kernel" less generic of a name?)
                            V.kernel.args.make_inplace(
                                input_node.get_name(), self.get_name()
                            )
                            # mutations not tracked in cpp kernels
                            if isinstance(
                                V.kernel, torch._inductor.codegen.triton.TritonKernel
                            ):
                                V.kernel.mutations.add(input_node.get_name())
                                V.kernel.mutations.add(self.get_name())
                        return

        V.graph.wrapper_code.codegen_allocation(self.node)

    def can_free(self):
        for use in self.users:
            if isinstance(use.node, OutputNode):
                return False
        return True

    def codegen_originating_info(self, buffer, only_once=True):
        if not config.comment_origin:
            return

        if only_once and self.written:
            return
        origins = self.node.origins
        out_lines = []

        for o in origins:
            if o.op == "output":
                # These are boring and samey
                continue

            out_lines.append("")
            # TODO(voz): Should the pragma be constant somewhere?
            out_lines.append("#pragma CMT ORIGIN:")
            out_lines.append(f"#pragma CMT {o.op} {o.target}")
            if "stack_trace" in o.meta:
                stack_trace = f"{o.meta['stack_trace']}"
                stack_trace_last_line = stack_trace.split("|")[-1]
                out_lines.append(
                    "#pragma CMT "
                    + stack_trace_last_line.replace("{", "{{")
                    .replace("}", "}}")
                    .replace("\n", "\\")
                )
                out_lines.append("#pragma CMT END ORIGIN")
                out_lines.append("")

        if len(out_lines) == 0:
            return

        # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
        # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
        buffer.writelines(out_lines)
        self.written = True


class ExternKernelSchedulerNode(BaseSchedulerNode):
    def debug_str_extra(self):
        return f"{self.get_name()}.node.kernel = {getattr(self.node, 'kernel', None)}"

    def is_extern(self):
        return True

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        if self.get_aliases() or self.is_template():
            return False

        if read_dep.name not in self.scheduler.name_to_node:
            # don't allow reuse of an 'input' buffer, we don't own it
            # (would this have been fixed if I tracked mutations properly above?)
            return False

        if not isinstance(self.node, torch._inductor.ir.AllReduce):
            # TODO make this a property of the IR
            return False

        if len(self.read_writes.writes) == 1:
            write_dep = next(iter(self.read_writes.writes))
            return read_dep.numbytes_hint() == write_dep.numbytes_hint()

        return False


class NopKernelSchedulerNode(BaseSchedulerNode):
    pass


class SchedulerNode(BaseSchedulerNode):
    def __init__(
        self,
        scheduler: "Scheduler",
        node: ir.ComputedBuffer,
        group_fn,
        loop_reorder=None,
    ):
        super().__init__(scheduler, node)

        if loop_reorder is None:

            def loop_reorder(n):
                return n.simplify_and_reorder()

        else:
            assert (
                not self.is_reduction()
            ), "loop_reorder is not supported for reductions"

        self._sizes, self._body = loop_reorder(node)

        self.group = (node.get_device(), group_fn(self._sizes))

        if self.is_template():
            self.set_read_writes(node.normalized_read_writes())
        else:
            self.set_read_writes(
                dependencies.extract_read_writes(
                    self._body, *self._sizes, normalize=True
                )
            )

        if self.is_reduction():
            # reduction has last (reduced) dim in its sizes, and some
            # downstream dependencies get confused by it
            self.read_writes.writes = self.read_writes.writes | {
                w.strip_last_size() for w in self.read_writes.writes
            }

    def debug_str_extra(self):
        name = self.get_name()
        lines = [
            f"{name}.group.device = {self.group[0]}",
            f"{name}.group.iteration = {self.group[1]}",
            f"{name}.sizes = {self._sizes}",
        ]
        if self.get_aliases():
            lines.append(f"{name}.aliases = {pformat(self.get_aliases())}")
        if self.get_mutations():
            lines.append(f"{name}.mutations = {pformat(self.get_mutations())}")
        if isinstance(self._body, ir.LoopBody):
            lines.append(f"class {name}_loop_body:")
            lines.append(textwrap.indent(self._body.debug_str(), "    "))
        return "\n".join(lines)

    def get_ranges(self):
        return self._sizes

    def is_reduction(self):
        return bool(self.node.get_reduction_type())

    def is_template(self):
        return isinstance(self.node, ir.TemplateBuffer)

    def run(self, *index_vars):
        self.mark_run()
        self.codegen(index_vars)

    def mark_run(self):
        self.allocate()

    def ranges_from_index_vars(self, index_vars):
        sizes = self._sizes
        assert sum(map(len, sizes)) == sum(map(len, index_vars))
        var_ranges = dict(
            zip(
                itertools.chain.from_iterable(index_vars),
                itertools.chain.from_iterable(sizes),
            )
        )
        return var_ranges

    def codegen(self, index_vars):
        var_ranges = self.ranges_from_index_vars(index_vars)
        try:
            with V.set_ops_handler(
                SimplifyIndexing(V.get_ops_handler(), var_ranges)
            ), V.kernel.set_current_node(self):
                self._body(*index_vars)
        except Exception:
            log.fatal("Error in codegen for %s", self.node)
            raise

    def pointwise_read_writes(self):
        """
        Get the memory dependencies in the non-reduction axis.
        """
        sizes, reduction_sizes = self._sizes

        def fn(index):
            return self._body(index, [sympy.Integer(0) for _ in reduction_sizes])

        return dependencies.extract_read_writes(fn, sizes)

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        if self.get_aliases() or self.is_template():
            return False
        if len(self.read_writes.writes) == 1 and hasattr(read_dep, "index"):
            write_dep = next(iter(self.read_writes.writes))
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False

    def reindex_loop(self, reindex: Callable, new_size: List[sympy.Expr]) -> "SchedulerNode":
        """Returns a new node which represents the same loop iterated in a new order

        reindex is a callable taking a new index and returning the corresponding original index.
        new_size is the new ranges of the index variables.

        Also note some class members are shallow copied to the new node, so do
        not modify the returned node.

        """

        assert reindex(new_size) == self._sizes[0]

        def loop_reorder(ir_node: ir.ComputedBuffer):
            new_sizes = new_size, self._sizes[1]
            (
                iter_vars,
                reduce_vars,
            ), var_ranges = dependencies.index_vars_no_squeeze(
                *new_sizes, prefix="reindex"
            )
            assert len(reduce_vars) == 0
            body = ir.LoopBody(
                self._body,
                [reindex(iter_vars), reduce_vars],
                var_ranges,
            )
            return new_sizes, body

        new_node = SchedulerNode(
            self.scheduler,
            self.node,
            self.scheduler.get_backend(self.get_device()).group_fn,
            loop_reorder=loop_reorder,
        )
        new_node.users = self.users
        new_node.inverse_users = self.inverse_users
        new_node.recursive_predecessors = self.recursive_predecessors
        new_node.min_order = self.min_order
        new_node.max_order = self.max_order
        new_node.last_usage = self.last_usage
        new_node.written = self.written
        return new_node


class FusedSchedulerNode(BaseSchedulerNode):
    """
    This is a "fake" scheduler node that represents a group of scheduler nodes
    that are meant to be fused together. The way it does this is by maintaining
    its unmet dependencies as the union of its constituent nodes.
    """

    @classmethod
    def fuse(cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        assert node1.scheduler is node2.scheduler
        return cls(node1.scheduler, node1.get_nodes() + node2.get_nodes())

    def __init__(self, scheduler: "Scheduler", snodes: List[SchedulerNode]):
        # NB: No need to call super().__init__() because we don't need to re-use any of its logic.
        self.snodes = snodes
        self.scheduler = scheduler
        self.node = None  # type: ignore[assignment]
        self.users = None
        self.inverse_users = []
        self.group = max(snodes, key=lambda x: int(x.is_reduction())).group
        self.recursive_predecessors = functools.reduce(
            set.union, [x.recursive_predecessors for x in snodes]
        )
        self.set_read_writes(
            functools.reduce(
                dependencies.ReadWrites.merge, [x.read_writes for x in snodes]
            )
        )
        names = set(self.get_names())
        self.unmet_dependencies = {
            dep
            for dep in functools.reduce(
                set.union, [x.unmet_dependencies for x in snodes]
            )
            if dep.name not in names
        } - self.read_writes.writes
        self.min_order = min([x.min_order for x in self.snodes])
        self.max_order = max([x.max_order for x in self.snodes])

    @cache_on_self
    def get_name(self) -> str:
        return "_".join([x.get_name() for x in self.snodes])

    def get_first_name(self) -> str:
        return self.snodes[0].get_name()

    @cache_on_self
    def get_names(self) -> Set[str]:
        return functools.reduce(set.union, [x.get_names() for x in self.snodes])

    def debug_str_extra(self):
        return (
            f"{self.get_name()}.snodes = {pformat([x.get_name() for x in self.snodes])}"
        )

    @cache_on_self
    def used_buffer_names(self) -> Set[str]:
        return functools.reduce(set.union, [x.used_buffer_names() for x in self.snodes])

    def get_nodes(self) -> List[BaseSchedulerNode]:
        return self.snodes

    def __repr__(self):
        return f"{type(self).__name__}(nodes={self.get_name()})"

    @cache_on_self
    def is_reduction(self):
        return any(x.is_reduction() for x in self.snodes)

    @cache_on_self
    def is_template(self):
        return any(x.is_template() for x in self.snodes)

    def get_device(self):
        return self.group[0]

    @cache_on_self
    def has_aliasing_or_mutation(self):
        return any(x.has_aliasing_or_mutation() for x in self.snodes)

    # None of these need to be implemented, as a FusedSchedulerNode is just an
    # abstraction for scheduling purposes
    def update_mutated_names(self, renames: Dict[str, str]):
        raise NotImplementedError

    def add_mutation_dep(self, name):
        raise NotImplementedError

    def set_users(self, users: List["NodeUser"]):
        raise NotImplementedError

    def get_aliases(self):
        raise NotImplementedError

    def get_mutations(self):
        raise NotImplementedError

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        raise NotImplementedError

    def allocate(self):
        raise NotImplementedError

    def can_free(self):
        raise NotImplementedError

def reindex_uncoalesce_to(original_shape: List[sympy.Expr], target_shape: List[sympy.Expr]):
    """
    """
    target_idx = 0
    dim_offset = 0
    size_hint = V.graph.sizevars.size_hint
    reindex = None
    for dim, length in enumerate(original_shape):
        cur_size = target_shape[dim + dim_offset]
        if length == cur_size:
            continue

        sizes = [cur_size]
        while size_hint(cur_size) < size_hint(length):
            ++dim_offset
            target_len = target_shape[dim + dim_offset]
            cur_size *= target_len
            sizes.append(target_len)

        assert cur_size == length, (cur_size, length, dim, original_shape, target_shape)

        dim_reindex = ir.reindex_split_dim(dim + dim_offset, sizes)
        dim_offset += len(sizes) - 1

        reindex = ir.maybe_fuse_reindexing(reindex, dim_reindex)

    if reindex is None:
        return lambda index: index
    return reindex


class ReorderedSchedulerNode:
    def __init__(
            self,
            node: Union[SchedulerNode, FusedSchedulerNode],
            order: List[int],
            original_shape: List[int],
    ):
        self.node = node
        self.order = order
        self.original_shape = original_shape


    @staticmethod
    def _reorder_node(node: BaseSchedulerNode, order: List[int], shape: List[sympy.Expr]) -> BaseSchedulerNode:
        assert len(order) == len(shape)
        if isinstance(node, FusedSchedulerNode):
            new_nodes = [
                ReorderedSchedulerNode._reorder_node(n, order, shape)
                for n in node.get_nodes()
            ]
            return FusedSchedulerNode(node.scheduler, new_nodes)

        assert isinstance(node, SchedulerNode)
        cur_shape = node.get_ranges()[0]
        print(cur_shape, shape)
        uncoalesce = (reindex_uncoalesce_to(cur_shape, shape)
                      if cur_shape != shape else None)

        reorder = ir.same_reorder(order)
        inv_reorder = ir.inverse_reorder(order)
        new_shape = inv_reorder(shape)
        reindex = ir.maybe_fuse_reindex(reorder, uncoalesce)
        return node.reindex_loop(reindex, new_shape)

    @staticmethod
    def reorder_node(
        node: BaseSchedulerNode, order: List[int], shape: List[sympy.Expr]
    ) -> "ReorderedSchedulerNode":
        new_node = ReorderedSchedulerNode._reorder_node(node, order, shape)
        wrapped = ReorderedSchedulerNode(new_node, order, shape)
        print(f"Reordering {node} into {wrapped}")
        return wrapped

    def reapply(self, node: BaseSchedulerNode) -> "ReorderedSchedulerNode":
        return ReorderedSchedulerNode.reorder_node(node, self.order, self.shape)

    def __repr__(self):
        return f"{type(self).__name__}(nodes={self.get_names()}, order={self.order}, original_shape={self.original_shape})"

    def __getattr__(self, item):
        return getattr(self.node, item)

def pick_loop_order(stride_lengths, sizes, priority_idx=()):
    """
    A heuristic to decide loop iteration orders.  This has not been well
    tuned and may be something we should autotune.
    """

    @functools.cmp_to_key
    def index_cmp(a, b):
        if sizes[a] == 1 or sizes[b] == 1:
            # 1-sizes don't matter, just move them to the end
            return cmp(sizes[a] == 1, sizes[b] == 1)

        stride_len_a = [sl[a] for sl in stride_lengths]
        stride_len_b = [sl[b] for sl in stride_lengths]

        # equivalent to
        # np.logical_or(stride_lengths[:, b] == 0, stride_lengths[:, a] < stride_lengths[:, b]).all()
        a_first = all(
            sl_b == 0 or sl_a < sl_b for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        b_first = all(
            sl_a == 0 or sl_b < sl_a for sl_a, sl_b in zip(stride_len_a, stride_len_b)
        )
        if a_first and not b_first:
            return -1
        if b_first and not a_first:
            return 1

        # otherwise contiguous
        return cmp(b, a)

    order = list(reversed(range(len(stride_lengths[0]))))
    if len(priority_idx) > 0:
        # if we have priority node, only use that node's order
        stride_lengths = [stride_lengths[pi] for pi in priority_idx]
    if config.pick_loop_orders:
        order.sort(key=index_cmp)
    return order


@dataclasses.dataclass
class NodeUser:
    node: BaseSchedulerNode
    can_inplace: bool = False

    def get_name(self):
        return self.node.get_name()


class Scheduler:
    @dynamo_timed
    def __init__(self, nodes):
        super().__init__()
        self.backends = {}

        self.nodes = []
        self.available_buffer_names = {
            *V.graph.graph_inputs.keys(),
            *V.graph.constants.keys(),
        }
        for node in nodes:
            assert (
                node.origins is not None
            ), "All nodes passed to scheduling must have an origin"
            if node.is_no_op():
                self.nodes.append(NopKernelSchedulerNode(self, node))
            elif isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
                group_fn = self.get_backend(node.get_device()).group_fn
                self.nodes.append(SchedulerNode(self, node, group_fn))
            elif isinstance(node, ir.ExternKernel):
                self.nodes.append(ExternKernelSchedulerNode(self, node))
            else:
                raise NotImplementedError(node)
        # some new constants could have been created above
        self.available_buffer_names.update(V.graph.constants.keys())
        for node in self.nodes:
            node.prune_deps()

        self.name_to_node = {node.get_name(): node for node in self.nodes}
        self.name_to_fused_node = None  # set in fuse_nods()

        # we handle mutation by renaming modified versions of the same
        # buffer in the dependency graph to prevent cycles.
        # mutation_renames: tracks the current name for a given buffer
        #                   (changed once per mutation)
        self.mutation_real_name = {}
        # mutation_real_name: maps back to the original name for codegen
        self.mutation_renames = {}

        self.compute_dependencies()
        self.topological_sort_schedule()
        self.compute_predecessors()
        self.dead_node_elimination()

        metrics.ir_nodes_pre_fusion += len(self.nodes)
        V.debug.ir_pre_fusion(self.nodes)
        self.num_orig_nodes = len(self.nodes)
        self.name_to_fused_node = {n.get_name(): n for n in self.nodes}
        self.fuse_nodes()
        self.compute_last_usage()
        V.debug.ir_post_fusion(self.nodes)
        V.debug.graph_diagram(self.nodes)
        self.debug_draw_graph()

        # used during codegen:
        self.current_device = None
        self.buffer_names_to_free = set()
        self.buffer_names_no_longer_needed = set()

    def debug_draw_graph(self):
        """Generate an image of the graph for debugging"""
        if os.environ.get("INDUCTOR_WRITE_SCHEDULER_GRAPH", None) == "1":
            from .debug import draw_buffers

            draw_buffers(self.nodes, print_graph=True)

    def debug_print_nodes(self, label):
        if log.isEnabledFor(logging.INFO):
            log.info("%s:", label)
            for node in self.nodes:
                node.log_details()

    def compute_dependencies(self):
        """
        Create dependency edges between nodes, handling aliasing and
        mutation properly.
        """
        name_to_users = collections.defaultdict(list)

        # handle aliasing by using python aliasing in name_to_users
        # if foo aliases bar then we will make name_to_users["foo"] point
        # to the same python list as name_to_users["bar"]
        for node1 in self.nodes:
            node1_name = node1.get_name()
            for node2_name in node1.get_aliases():
                if node1_name in name_to_users and node2_name in name_to_users:
                    # merge the two
                    list1 = name_to_users[node1_name]
                    list2 = name_to_users[node2_name]
                    combined = list1 + list2
                    for key in name_to_users.keys():
                        if name_to_users[key] is list1 or name_to_users[key] is list2:
                            name_to_users[key] = combined
                elif node1_name in name_to_users:
                    name_to_users[node2_name] = name_to_users[node1_name]
                else:
                    name_to_users[node1_name] = name_to_users[node2_name]

        def rename(n):
            if n in self.mutation_renames:
                return rename(self.mutation_renames[n])
            return n

        def dep_closure(node_name):
            reachable_names = {node_name}
            node = self.name_to_node[node_name]
            write_dep = list(node.read_writes.writes)[0]
            for read_dep in node.read_writes.reads:
                if (
                    read_dep.name in self.name_to_node
                    and read_dep.index == write_dep.index
                    and read_dep.size == write_dep.size
                ):
                    reachable_names.update(dep_closure(read_dep.name))
            return reachable_names

        def add_user(used_by_name, user_node, can_inplace=False):
            name_to_users[rename(used_by_name)].append(NodeUser(user_node, can_inplace))

        for node in self.nodes:
            # a node will mutate either 0 or 1 buffers
            for alt_name in node.get_mutations():
                alt_name = rename(alt_name)
                # this node must run after the prior writer
                add_user(alt_name, node)
                node.add_mutation_dep(StarDep(alt_name))
                for other_node in name_to_users[alt_name]:
                    # this node must run after all prior readers
                    other_name = rename(other_node.get_name())
                    known_dep_node_names = dep_closure(node.get_name())
                    if other_name not in known_dep_node_names:
                        # If this node already directly or indirectly depends on other_node,
                        # we don't need to insert an extra dep.
                        node.add_mutation_dep(WeakDep(other_name))
                        add_user(other_name, node)

            # add normal non-mutation dependencies
            for read in node.read_writes.reads:
                add_user(read.name, node, node.can_inplace(read))

            node.update_mutated_names(self.mutation_renames)

            # update our renaming scheme for the next iteration
            for alt_name in node.get_mutations():
                self.mutation_renames[rename(alt_name)] = node.get_name()
                self.mutation_renames[alt_name] = node.get_name()
                self.mutation_real_name[node.get_name()] = self.mutation_real_name.get(
                    alt_name, alt_name
                )

        # make sure outputs aren't dead-code-eliminated
        for node_name in V.graph.get_output_names():
            add_user(node_name, OutputNode(StarDep(node_name)))

        # make sure input mutation isn't dead-code-eliminated
        for name in self.mutation_renames:
            if name in V.graph.graph_inputs:
                add_user(name, OutputNode(StarDep(name)))
                V.graph.mutated_inputs.add(name)

        # copy users information onto the nodes
        for node in self.nodes:
            node.set_users(name_to_users[node.get_name()])

        # populate inverse_users
        for node in self.nodes:
            for user in node.users:
                user.node.inverse_users.append(node)

    def dead_node_elimination(self):
        """
        Remove any nodes without users
        """
        updated_nodes = []
        for node in self.nodes:
            if node.users:
                updated_nodes.append(node)
            else:
                # dead code
                log.debug("removed dead node: %s", node.get_name())
                V.graph.removed_buffers.add(node.get_name())
        self.nodes = updated_nodes

    def topological_sort_schedule(self):
        """
        Ensure self.nodes is in topologically sorted order
        """
        seen = set()
        name_to_node = dict()
        result = []

        def visit(n):
            if n not in seen:
                seen.add(n)
                for dep in sorted(n.unmet_dependencies, key=lambda d: d.name):
                    visit(name_to_node[dep.name])
                result.append(n)

        for node in self.nodes:
            for name in node.get_names():
                name_to_node[name] = node
        for node in self.nodes:
            visit(node)
        self.nodes = result

    def compute_predecessors(self):
        """
        Populate each node.recursive_predecessors
        """
        # note self.nodes is topologically sorted
        name_to_predecessors = {}
        for node in self.nodes:
            recursive_predecessors = set()
            for dep in node.unmet_dependencies:
                recursive_predecessors.add(dep.name)
                recursive_predecessors |= name_to_predecessors[dep.name]
            name_to_predecessors[node.get_name()] = recursive_predecessors
            node.recursive_predecessors = recursive_predecessors

        for order, node in enumerate(self.nodes):
            node.min_order = order
            node.max_order = order

    def fuse_nodes(self):
        """
        Mutates self.nodes to combine nodes into FusedSchedulerNodes.
        """
        for i in range(10):
            old_len = len(self.nodes)
            self.fuse_nodes_once()
            if len(self.nodes) == old_len:
                print(f"No fusions after {i} iterations")
                break

    def fuse_nodes_once(self):
        """
        Mutates self.nodes to combine nodes into FusedSchedulerNodes.

        This relies on two key functions to control the logic:
            - self.can_fuses(): checks if a fusion is legal
            - self.score_fusion(): assigns priority to a given fusion
        """
        fused_nodes = set(self.nodes)

        # recent_fusions tracks names of nodes fused during this function call.
        #
        # This is required because config.fusion_reorder_loops, when enabled,
        # causes get_possible_fusions() to return new nodes not in
        # name_to_fused_node, so we can't assume
        #   node = self.name_to_fused_node[node.get_first_name()]
        # will map an unfused nodes back to itself
        recent_fusions: Set[str] = set()
        any_reordered = False

        p = self.get_possible_fusions()

        for node1, node2 in p:
            nodes = (node1, node2)

            def update_node(n):
                order = None
                name = n.get_first_name()
                if isinstance(n, ReorderedSchedulerNode):

                    order = n.order
                    n = n.node

                name = n.get_first_name()
                if name not in recent_fusions:
                    return n

                fused_node = self.name_to_fused_node[n.get_first_name()]
                if isinstance(n, ReorderedSchedulerNode):
                    return n.reapply(fused_node)

                return fused_node

            node1 = update_node(node1)
            if node1 is None:
                continue
            node2 = update_node(node2)
            if node2 is None:
                continue

            is_reordered = (
                    isinstance(nodes[0], ReorderedSchedulerNode) or
                    isinstance(nodes[1], ReorderedSchedulerNode)
            )
            if is_reordered:
                print(f"Attempting to fuse reordered nodes {node1.get_names()}, {node2.get_names()}")

            if self.can_fuse(node1, node2) and not self.will_fusion_create_cycle(
                node1, node2
            ):
                if is_reordered:
                    print("Fusing reordered nodes")
                    print(nodes[0])
                    print(nodes[1])
                node3 = FusedSchedulerNode.fuse(node1, node2)
                fused_nodes.remove(self.name_to_fused_node[node1.get_first_name()])
                fused_nodes.remove(self.name_to_fused_node[node2.get_first_name()])
                fused_nodes.add(node3)
                self.name_to_fused_node.update(
                    {n.get_name(): node3 for n in node3.get_nodes()}
                )

                recent_fusions.add(node1.get_first_name())
                recent_fusions.add(node2.get_first_name())

        self.nodes = sorted(fused_nodes, key=lambda x: x.min_order)
        self.topological_sort_schedule()
        self.prune_redundant_deps()

    def prune_redundant_deps(self):
        for node in self.nodes:
            node.prune_redundant_deps(self.name_to_fused_node)

    def get_possible_fusions(self):
        """
        Helper to find all legal fusion opportunities, sorted by self.score_fusion()
        """
        possible_fusions = []
        seen = set()

        def check_all_pairs(nodes):
            for node1, node2 in itertools.combinations(nodes, 2):
                key = (node1, node2)
                if key in seen:
                    continue
                seen.add(key)

                if self.can_fuse(node1, node2):
                    possible_fusions.append(key)
                elif node2.is_template() and self.can_fuse(node2, node1):
                    # epilogue fusions are order dependent
                    possible_fusions.append((node2, node1))

        buffer_names_grouping = collections.defaultdict(list)
        for node in self.nodes:
            for buf in node.used_buffer_names():
                buffer_names_grouping[buf].append(node)
        for node_grouping in buffer_names_grouping.values():
            check_all_pairs(node_grouping)

        if config.fusion_reorder_loops:
            reorder_fusions = self.get_reorder_fusions(
                buffer_names_grouping, ignore=set(possible_fusions)
            )
            print(f"{len(reorder_fusions)} reorder opportunities")
            possible_fusions += reorder_fusions

        if config.aggressive_fusion:
            group_grouping = collections.defaultdict(list)
            for node in self.nodes:
                group = getattr(node, "group", None)
                if group:
                    group_grouping[group].append(node)
            for node_grouping in group_grouping.values():
                check_all_pairs(node_grouping)

        return sorted(possible_fusions, key=self.score_fusion_key, reverse=True)

    @staticmethod
    def pick_dependency_reorder(
        dep1: MemoryDep, dep2: MemoryDep
    ) -> Optional[List[int]]:
        """Checks if dep2's iteration order can be permuted to make it match dep1, and
        if so returns the new permutation.
        """
        if dep1 == dep2:
            return None

        sizevars = V.graph.sizevars

        # Step 1: Can we reorder to match their shapes?
        def sort_exprs(exprs):
            return sorted(exprs, key=lambda e: e.sort_key())

        if sort_exprs(dep1.size) != sort_exprs(dep2.size):
            return None

        # Step 2: Can we reorder to match their strides?
        ndim = len(dep1.size)
        c = canonicalization_prefix()
        index_vars = [sympy_symbol(f"{c}{i}") for i in range(ndim)]
        strides1 = sizevars.stride_vars(dep1.index, index_vars)
        strides2 = sizevars.stride_vars(dep2.index, index_vars)

        dep1_permutation = sorted(range(ndim), key=lambda i: strides1[i].sort_key())
        dep2_permutation = sorted(range(ndim), key=lambda i: strides2[i].sort_key())
        if dep1_permutation == dep2_permutation:
            return None

        valid = all(
            dep1.size[d1] == dep2.size[d2]
            for d1, d2 in zip(dep1_permutation, dep2_permutation)
        )
        if not valid:
            return None

        # Sanity check, if we apply the reordering to the index expression, it should be equal
        # This may fail for non-strided index expressions
        dep2_reindexed = sympy_subs(
            dep2.index,
            {
                index_vars[d2]: index_vars[d1]
                for d1, d2 in zip(dep1_permutation, dep2_permutation)
            },
        )
        if dep1.index != dep2_reindexed:
            return None

        # Permute node2 to iterate in the same order as node1
        mapping = dict(zip(dep2_permutation, dep1_permutation))
        reorder = [mapping[i] for i in range(ndim)]

        assert ir.same_reorder(reorder)(strides1) == strides2, \
            (strides1, strides2, reorder, dep1_permutation, dep2_permutation)
        return reorder

    def get_reorder_fusions(
        self,
        buffer_names_grouping: Dict[str, List[BaseSchedulerNode]],
        ignore: Set[Tuple[BaseSchedulerNode, BaseSchedulerNode]],
    ) -> Set[Tuple[BaseSchedulerNode, BaseSchedulerNode]]:
        """Returns possible fusions that require iterating an ir.Loop in a different
        order. To do this, we create new nodes that represent the reordered
        Loop. Any pair of nodes in the ignore set will not be checked, on the
        assumption these are better to fuse without reordering.
        """
        possible_fusions = set()

        # Step 1: Find pairs of nodes that could possibly be fused after reordering
        #
        # The strategy is to compare nodes pairs that access a common buffer
        # and compare their `read_writes`. If the MemoryDep objects for the
        # common buffer aren't equal then the two nodes must iterate the buffer
        # in different orders. So, we inspect the MemoryDep's indexing
        # expressions to find a reordering that makes them equal.
        #
        # Note that since nodes can have multiple buffers in common, this may
        # produce duplicate reorderings so we build the full set of reorderings
        # first to eliminate duplicates.
        reorder_opportunities: Dict[
            Tuple[BaseSchedulerNode, BaseSchedulerNode], Set[Tuple[int, ...]]
        ] = collections.defaultdict(set)

        for buf_name, node_grouping in buffer_names_grouping.items():
            for node1, node2 in itertools.combinations(node_grouping, 2):
                if (node1, node2) in ignore:
                    continue

                if node1.is_reduction() and node2.is_reduction():
                    continue

                if any(
                    not isinstance(n, (FusedSchedulerNode, SchedulerNode))
                    for n in itertools.chain(node1.get_nodes(), node2.get_nodes())
                ):
                    continue

                def node_ndim(node):
                    pointwise, reduction = node.get_ranges()
                    return len(pointwise) + len(reduction)

                ndim = max(node_ndim(n) for n in node1.get_nodes())

                def filter_in(dep):
                    if not (
                        isinstance(dep, MemoryDep)
                        and dep.name == buf_name
                    ):
                        return False
                    if len(dep.size) == ndim:
                        return True
                    ranges = (
                        [n.get_ranges() for n in node1.get_nodes()] +
                        [n.get_ranges() for n in node2.get_nodes()])
                    return False

                node1_buf_deps = [
                    dep
                    for dep in (node1.read_writes.reads | node1.read_writes.writes)
                    if filter_in(dep)
                ]
                node2_buf_deps = [
                    dep
                    for dep in (node2.read_writes.reads | node2.read_writes.writes)
                    if filter_in(dep)
                ]

                for dep1, dep2 in itertools.product(node1_buf_deps, node2_buf_deps):
                    if dep1 == dep2:
                        continue

                    reorder = self.pick_dependency_reorder(dep1, dep2)
                    if reorder is not None:
                        assert reorder != list(range(len(reorder))), (reorder, dep1, dep2)
                        reorder_opportunities[(node1, node2)].add((tuple(reorder), dep1.size))

        # Step 2: Construct new nodes with the reordering and comprehesively
        # test if they can be fused with self.can_fuse
        for (node1, node2), orders in reorder_opportunities.items():
            for order, size in orders:
                # Prefer to reorder pointwise ops, or node2 if both match
                reorder_node1 = node2.is_reduction() and not node1.is_reduction()
                if reorder_node1:
                    assert isinstance(node1, (SchedulerNode, FusedSchedulerNode))
                    inv_order = ir.invert_permutation(order)
                    new_node = ReorderedSchedulerNode.reorder_node(node1, inv_order, list(size))
                    node_pair = (new_node, node2)
                else:
                    assert isinstance(node2, (SchedulerNode, FusedSchedulerNode))
                    old_size = ir.same_reorder(order)(size)
                    new_node = ReorderedSchedulerNode.reorder_node(node2, order, old_size)
                    node_pair = (node1, new_node)

                if self.can_fuse(*node_pair):
                    print(f"Possible reordering {node1} {node2} {order}")
                    possible_fusions.add(node_pair)
                else:
                    print(f"Cannot fuse after reordering {node1} {node2} {order}")
                    print(f"{node_pair[0]}")
                    print(f"{node_pair[1]}")

        return possible_fusions

    def will_fusion_create_cycle(self, node1, node2):
        """Finds whether there's a path from src to dst caused indirectly by fusion"""

        def check(node):
            if isinstance(node, FusedSchedulerNode) and node not in visited:
                visited.add(node)
                return bool(combined_names & node.recursive_predecessors) or any(
                    check(self.name_to_fused_node[n])
                    for n in node.recursive_predecessors - combined_predecessors
                )
            return False

        visited = set()
        combined_names = node1.get_names() | node2.get_names()
        combined_predecessors = (
            node1.recursive_predecessors | node2.recursive_predecessors
        ) - combined_names
        return any(check(self.name_to_fused_node[n]) for n in combined_predecessors)

    def can_fuse(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        """
        Determine if it is possible to combine node1 and node2 into a
        single fused node.
        """
        if node1 is node2:
            return False
        if (
            isinstance(node1, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node1.is_template()
        ):
            return False
        if (
            isinstance(node2, (ExternKernelSchedulerNode, NopKernelSchedulerNode))
            and not node2.is_template()
        ):
            return False
        if node2.get_names() & node1.recursive_predecessors:
            return False  # node2 must go before node1
        if node2.is_template():
            return False  # only epilogues
        if node1.is_template() and (
            node2.has_aliasing_or_mutation()
            or node2.is_reduction()
            or not config.epilogue_fusion
        ):
            return False

        device = node1.get_device()
        if device != node2.get_device():
            return False  # wrong device

        # if (
        #         isinstance(node1, ReorderedSchedulerNode) or
        #         isinstance(node2, ReorderedSchedulerNode)
        # ):
        #     import pdb
        #     pdb.set_trace()

        no_shared_data = self.score_fusion_memory(node1, node2) == 0
        if no_shared_data and (
            not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
        ):
            log.info(f"Missed fusion of {node1} and {node2}: no shared data")
            return False  # heuristic not needed for correctness

        if len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size:
            log.info(f"Missed fusion of {node1} and {node2}: max fusion size")
            return False  # heuristic not needed for correctness

        if node1.get_names() & node2.recursive_predecessors:
            # node2 depends on node1 outputs
            if not self.can_fuse_vertical(node1, node2):
                return False
            return self.get_backend(device).can_fuse_vertical(node1, node2)
        else:  # nodes don't depend on each other, but may have common reads
            return self.get_backend(device).can_fuse_horizontal(node1, node2)

    def can_fuse_vertical(self, node1, node2):
        """
        Check if it is legal to fuse a consumer (node2) into a producer (node1).

        We can fuse them if all the reads of node2 either match
        corresponding writes in node1, or are written by nodes that can
        be scheduled before the fusion of node1 and node2.
        """
        node1_names = node1.get_names()
        computed_deps = set()

        for rd in node2.unmet_dependencies:
            for cd in node1.read_writes.writes:
                # StarDep doesn't match MemoryDep, different indices don't match
                # However, broadcasting sometimes strips dimensions, and if that's the case
                # we still can match unmet dep
                if (
                    rd.name == cd.name
                    and type(rd) == type(cd)
                    and rd.index == cd.index
                    and len(rd.size) >= len(cd.size)
                    and rd.size[: len(cd.size)] == cd.size
                ):
                    computed_deps.add(rd)

        remaining_deps = {dep.name for dep in node2.unmet_dependencies - computed_deps}
        if remaining_deps & node1_names:
            # MemoryDeps didn't match and read different locations of the same buffer.
            # Examples here include:
            #   - MemoryDep("foo", x) != MemoryDep("foo", x + 1)
            #   - MemoryDep("foo", x) != StarDep("foo")
            log.info(f"Missed fusion of {node1} and {node2} [vertical]: uncomputed dep")
            return False
        for name in remaining_deps:
            if node1_names & self.name_to_fused_node[name].recursive_predecessors:
                log.info(f"Missed fusion of {node1} and {node2} [vertical]: creates dep cycle")
                return False
        return True

    def score_fusion(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):
        """
        Assign a score (higher comes first) to the fusion of node1
        and node2.  When different fusions conflict with each other,
        this is the way we decide what order to run them in.

        Our current score is based on:
        - Estimate of the saved memory operations
        - Fusions closer together in original order
        """
        memory_score = self.score_fusion_memory(node1, node2)
        proximity_score = -max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )
        return (
            node1.is_template() == config.epilogue_fusion_first and memory_score > 0,
            node1.is_reduction() == node2.is_reduction() and memory_score > 0,
            memory_score,
            proximity_score,
        )

    def score_fusion_memory(self, node1, node2):
        """
        The first term in our fusion score that estimates number of saved memory operations.
        """
        common_memory_deps = (node1.read_writes.reads | node1.read_writes.writes) & (
            node2.read_writes.reads | node2.read_writes.writes
        )
        return sum(dep.numbytes_hint() for dep in common_memory_deps)

    def score_fusion_key(self, nodes):
        """
        Shim for list.sort(key=...)
        """
        node1, node2 = nodes
        return self.score_fusion(node1, node2)

    def compute_last_usage(self):
        """
        Populate node.last_usage
        """

        future_used_buffers = set()
        for node_name in V.graph.get_output_names():
            future_used_buffers.add(node_name)

        for node in reversed(self.nodes):
            used_buffers = node.used_buffer_names()
            used_buffers = {self.mutation_real_name.get(k, k) for k in used_buffers}
            node.last_usage = used_buffers - future_used_buffers
            future_used_buffers.update(used_buffers)

    def free_buffers(self):
        """Free any buffers that are no longer needed"""
        for name in sorted(self.buffer_names_to_free - V.graph.removed_buffers):
            if name in self.name_to_node:
                node = self.name_to_node[name]
                if node.can_free():
                    V.graph.wrapper_code.codegen_free(node.node)
            elif name in V.graph.graph_inputs:
                storage = V.graph.graph_inputs[name].data
                assert storage.is_input_buffer()
                V.graph.wrapper_code.codegen_free(storage.data)

        self.buffer_names_to_free.clear()

    def remove_kernel_local_buffers(self):
        """
        Any buffers that are both created and have a last use in the
        same kernel can be removed.
        """
        for name in V.kernel.store_buffer_names & self.buffer_names_no_longer_needed:
            if (
                name not in V.kernel.must_keep_buffers
                and name not in V.kernel.args.input_buffers
                and name not in self.mutation_renames
                and name not in self.mutation_real_name
            ):
                # For inplace buffers subject to remove, we don't actually
                # remove them but put them in a dedicated set. This simplifies
                # the life cycle management of inplace buffers.
                # This set is used to
                # 1) avoid unnecessary store in DeferredLine.
                # 2) avoid alias var definitions in kernel.
                if name in V.kernel.args.inplace_buffers:
                    V.graph.inplaced_to_remove.add(name)
                else:
                    self.remove_buffer(name)

    def remove_buffer(self, name):
        # Assign a special value instead of deleting the entry
        # because we still rely on output_buffers's length to
        # generate unique arg name.
        log.debug("remove_buffer(%r)", name)
        V.kernel.args.output_buffers[name] = "REMOVED"
        V.graph.removed_buffers.add(name)

    def flush(self):
        for backend in self.backends.values():
            backend.flush()
        self.free_buffers()

    def codegen_extern_call(self, scheduler_node: ExternKernelSchedulerNode):
        assert isinstance(scheduler_node, ExternKernelSchedulerNode)
        scheduler_node.allocate()
        node = scheduler_node.node
        node.codegen(V.graph.wrapper_code)
        self.free_buffers()

    def create_backend(self, device: torch.device):
        assert (
            device.type != "cuda" or device.index is not None
        ), f"{device} should have been normalized in lowering"
        V.graph.device_types.add(device.type)
        if device.type == "cpu":
            from .codegen.cpp import CppScheduling

            return CppScheduling(self)
        else:
            if not has_triton():
                device_props = torch.cuda.get_device_properties(device)
                if device_props.major < 7:
                    raise RuntimeError(
                        f"Found {device_props.name} which is too old to be supported by the triton GPU compiler, which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA capability {device_props.major}.{device_props.minor}"  # noqa: B950
                    )
                else:
                    raise RuntimeError(
                        "Cannot find a working triton installation. More information on installing Triton can be found at https://github.com/openai/triton"  # noqa: B950
                    )
            from .codegen.triton import TritonScheduling

            return TritonScheduling(self)

    def get_backend(self, device: torch.device):
        if device not in self.backends:
            self.backends[device] = self.create_backend(device)
        return self.backends[device]

    @dynamo_timed
    def codegen(self):
        for node in self.nodes:
            self.buffer_names_no_longer_needed.update(node.last_usage)

            if not isinstance(node, NopKernelSchedulerNode):
                device = node.get_device()
                if (
                    device != self.current_device
                    or node.is_extern()
                    or node.is_template()
                ):
                    self.flush()
                if device != self.current_device:
                    if device.type == "cuda":
                        if self.current_device and self.current_device.type == "cuda":
                            V.graph.wrapper_code.codegen_cuda_device_guard_exit()
                        assert device.index is not None, "device should have an index"
                        V.graph.wrapper_code.codegen_cuda_device_guard_enter(
                            device.index
                        )
                    elif self.current_device and self.current_device.type == "cuda":
                        V.graph.wrapper_code.codegen_cuda_device_guard_exit()
                    self.current_device = device

            self.buffer_names_to_free.update(node.last_usage)

            if node.is_template():
                node, *epilogue = node.get_nodes()
                self.get_backend(device).codegen_template(node, epilogue)
            elif node.is_extern():
                self.codegen_extern_call(node)
            elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
                self.get_backend(device).codegen_nodes(node.get_nodes())
            else:
                assert isinstance(node, NopKernelSchedulerNode)
                node.allocate()

            if config.triton.debug_sync_kernel:
                self.get_backend(device).codegen_sync()

            self.available_buffer_names.update(node.get_names())

        self.flush()
