#include <nsimd/nsimd-all.hpp>
#include <cstdlib>
#include <vector>

int main() {
  std::vector<float, nsimd::allocator<float> > v;

  v.clear();
  v.resize(100);

  v.clear();
  v.resize(100);
  v.resize(10000);

  v.clear();
  v.reserve(30);

  for (int i = 0; i < 1000; i++) {
    v.push_back(float(i));
  }
  if (v.size() != 1000) {
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < 500; i++) {
    v.pop_back();
  }
  if (v.size() != 500) {
    exit(EXIT_FAILURE);
  }

  return 0;
}
