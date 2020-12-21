#ifndef BENCHES_HPP
#define BENCHES_HPP

#include <limits>
#include <cmath>
#include <climits>

namespace nsimd {
namespace benches {

template <typename T>
double rand_sign() {
  if (std::is_unsigned<T>::value) {
    return 1.;
  } else {
    return (::rand() % 2) ? 1. : -1.;
  }
}

template <typename T>
T rand_bits(T min, T max = std::numeric_limits<T>::max()) {
  T r;
  do {
    int nbits = sizeof(T) * CHAR_BIT;
    u64 x = 0;
    for (int i = 0; i < nbits; ++i) {
      x |= u64(::rand() % 2) << i;
    }
    r = *((T*)&x);
  } while (r < min || r > max);
  return r;
}

template <typename T>
T rand_from(T min, T max = std::numeric_limits<T>::max()) {
  // From: http://c-faq.com/lib/randrange.html
  return T(double(min)
      + (double(::rand()) / (double(RAND_MAX) / (double(max) - double(min) + 1))));
}

template <typename T>
T rand_fp(T min, T max) {
  T r;
  if (std::isinf(min) && std::isinf(max)) {
    // For now, we're not using this method for random number
    //r = rand_bits<T>(min, max);
    r = rand_from<T>(-1000000, 1000000);
  } else {
    r = rand_from<T>(min, max);
  }
  return r;
}

template <typename T>
T rand(T min, T max = std::numeric_limits<T>::max()) {
  return rand_from<T>(min, max);
}

template <>
float rand<float>(float min, float max) {
  return rand_fp<float>(min, max);
}

template <>
double rand<double>(double min, double max) {
  return rand_fp<double>(min, max);
}

}
}

#endif
