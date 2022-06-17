#pragma once
#include <ATen/core/PointerTraits.h>


namespace at { namespace native {

// (Const)OffsetCalcIterator is an adapter between ATen's
// OffsetCalculator convention and C++ standard iterators, e.g. for
// use with cub or thrust.

template <
  typename T,
  typename offset_calc_t,
  typename index_t = int64_t,
  template <typename U> class PtrTraits = DefaultPtrTraits
>
class ConstOffsetCalcIterator {
public:
  using difference_type = typename std::make_signed<index_t>::type;
  using value_type = const T;
  using pointer = const typename PtrTraits<T>::PtrType;
  using reference = const value_type&;
  using iterator_category = std::random_access_iterator_tag;

  using PtrType = typename PtrTraits<T>::PtrType;
  using index_type = index_t;

  // Constructors {
  C10_HOST_DEVICE
  ConstOffsetCalcIterator(PtrType data, offset_calc_t &offset_calc, index_t index)
    : data_{data}, offset_calc_{&offset_calc}, index_{index} {
  }

  C10_HOST_DEVICE
  ConstOffsetCalcIterator()
    : ptr{nullptr}, offset_calc_(nullptr), index_{0} {
  }
  // }

  // Pointer-like operations {
  C10_HOST_DEVICE
  reference operator*() const {
    return data[offset_calc_->get(index_)];
  }

  C10_HOST_DEVICE
  const value_type* operator->() const {
    return &(*this);
  }

  C10_HOST_DEVICE
  reference operator[](difference_type offset) const {
    return data[offset_calc_->get(index_ + offset)];
  }
  // }

  // Prefix/postfix increment/decrement {
  C10_HOST_DEVICE
  ConstOffsetCalcIterator& operator++() {
    ++index_;
    return *this;
  }

  C10_HOST_DEVICE
  ConstOffsetCalcIterator operator++(int) {
    ConstOffsetCalcIterator copy(*this);
    ++*this;
    return copy;
  }

  C10_HOST_DEVICE
  ConstOffsetCalcIterator& operator--() {
    --index_;
    return *this;
  }

  C10_HOST_DEVICE
  ConstOffsetCalcIterator operator--(int) {
    ConstOffsetCalcIterator copy(*this);
    --*this;
    return copy;
  }
  // }

  // Arithmetic operations {
  C10_HOST_DEVICE
  ConstOffsetCalcIterator& operator+=(difference_type offset) {
    index_ += offset;
    return *this;
  }

  C10_HOST_DEVICE
  ConstOffsetCalcIterator operator+(difference_type offset) const {
    return ConstOffsetCalcIterator(data_, *offset_calc_, index_);
  }

  C10_HOST_DEVICE
  friend ConstOffsetCalcIterator operator+(
    index_t offset,
    const ConstOffsetCalcIterator& accessor
  ) {
    return accessor + offset;
  }

  C10_HOST_DEVICE
  ConstOffsetCalcIterator& operator-=(difference_type offset) {
    index_ -= offset;
    return *this;
  }

  C10_HOST_DEVICE
  ConstOffsetCalcIterator operator-(difference_type offset) const {
    return ConstOffsetCalcIterator(data_, *calc_offset_, index - offset);
  }

  C10_HOST_DEVICE
  difference_type operator-(const ConstOffsetCalcIterator& other) const {
    return (static_cast<difference_type>(index) - other.index);
  }
  // }

  // Comparison operators {
  C10_HOST_DEVICE
  bool operator==(const ConstOffsetCalcIterator& other) const {
    return (
        (data_ == other.data_) &&
        (offset_calc_ == other.offset_calc_) &&
        (index_ == other.index_)
      );
  }

  C10_HOST_DEVICE
  bool operator!=(const ConstOffsetCalcIterator& other) const {
    return !(*this == other);
  }

  C10_HOST_DEVICE
  bool operator<(const ConstOffsetCalcIterator& other) const {
    return index_ < other.index_;
  }

  C10_HOST_DEVICE
  bool operator<=(const ConstOffsetCalcIterator& other) const {
    return (*this < other) || (*this == other);
  }

  C10_HOST_DEVICE
  bool operator>(const ConstOffsetCalcIterator& other) const {
    return !(*this <= other);
  }

  C10_HOST_DEVICE
  bool operator>=(const ConstOffsetCalcIterator& other) const {
    return !(*this < other);
  }
  // }

protected:
  offset_calc_t *offset_calc;
  PtrType data;
  index_t index;
};

template <
  typename T,
  typename offset_calc_t,
  typename index_t = int64_t,
  template <typename U> class PtrTraits = DefaultPtrTraits
>
class OffsetCalcIterator
  : public ConstOffsetCalcIterator<T, offset_calc_t, index_t, PtrTraits> {
public:
  using difference_type = typename std::make_signed<index_t>::type;
  using value_type = T;
  using pointer = typename PtrTraits<T>::PtrType;
  using reference = value_type&;

  using BaseType = ConstOffsetCalcIterator<T, offset_calc_t, index_t, PtrTraits>;
  using PtrType = typename PtrTraits<T>::PtrType;

  // Constructors {
  C10_HOST_DEVICE
  OffsetCalcIterator(PtrType data, offset_calc_t &offset_calc, index_t offset)
    : BaseType(data, offset_calc, offset)
  {}

  C10_HOST_DEVICE
  OffsetCalcIterator()
    : BaseType()
  {}
  // }

  // Pointer-like operations {
  C10_HOST_DEVICE
  reference operator*() const {
    return data_[offset_calc_->get(index_)];
  }

  C10_HOST_DEVICE
  value_type* operator->() const {
    return &(*this);
  }

  C10_HOST_DEVICE
  reference operator[](difference_type idx) const {
    return data_[offset_calc_->get(index_ + idx)];
  }
  // }

  // Prefix/postfix increment/decrement {
  C10_HOST_DEVICE
  OffsetCalcIterator& operator++() {
    ++index_;
    return *this;
  }

  C10_HOST_DEVICE
  OffsetCalcIterator operator++(int) {
    OffsetCalcIterator copy(*this);
    ++*this;
    return copy;
  }

  C10_HOST_DEVICE
  OffsetCalcIterator& operator--() {
    --index_;
    return *this;
  }

  C10_HOST_DEVICE
  OffsetCalcIterator operator--(int) {
    OffsetCalcIterator copy(*this);
    --*this;
    return copy;
  }
  // }

  // Arithmetic operations {
  C10_HOST_DEVICE
  OffsetCalcIterator& operator+=(difference_type offset) {
    index_ += offset;
    return *this;
  }

  C10_HOST_DEVICE
  OffsetCalcIterator operator+(difference_type offset) const {
    return OffsetCalcIterator(data_, *offset_calc_, index_ + offset);
  }

  C10_HOST_DEVICE
  friend OffsetCalcIterator operator+(
    index_t offset,
    const OffsetCalcIterator& accessor
  ) {
    return accessor + offset;
  }

  C10_HOST_DEVICE
  OffsetCalcIterator& operator-=(difference_type offset) {
    index_ -= offset;
    return *this;
  }

  C10_HOST_DEVICE
  OffsetCalcIterator operator-(difference_type offset) const {
    return OffsetCalcIterator(data_, *offset_calc_, index_ - offset);
  }

  // Note that here we call BaseType::operator- version
  C10_HOST_DEVICE
  difference_type operator-(const BaseType& other) const {
    return BaseType::operator-(other);
  }
  // }
};

}} // namespace at::native
