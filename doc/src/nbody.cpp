// Copyright (c) 2019 Agenium Scale
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <nsimd/nsimd-all.hpp>

template <typename T, typename A> void print(std::vector<T, A> const &v) {
  std::cout << "{ ";
  for (size_t i = 0; i < v.size(); ++i) {
    std::cout << (i == 0 ? "" : ", ") << v[i];
  }
  std::cout << " }";
}

template <typename T> struct particles_t {
  size_t N;

  std::vector<T> x;
  std::vector<T> y;
  std::vector<T> z;

  std::vector<T> m;

  std::vector<T> vx;
  std::vector<T> vy;
  std::vector<T> vz;

  particles_t(size_t N = 0)
      : N(N), x(N), y(N), z(N), m(N), vx(N), vy(N), vz(N) {}

  size_t size() const { return N; }
};

template <typename T> void nbody_step_scalar(particles_t<T> &particles) {
  T epsilon = 0.00125f;
  for (std::size_t i = 0; i < particles.size(); ++i) {
    T ax{0};
    T ay{0};
    T az{0};

    T pix = particles.x[i];
    T piy = particles.y[i];
    T piz = particles.z[i];

    T pim = particles.m[i];

    for (size_t j = i + 1; j < particles.size(); ++j) {
      T pjx = particles.x[j];
      T pjy = particles.y[j];
      T pjz = particles.z[j];

      T dx = pjx - pix;
      T dy = pjy - piy;
      T dz = pjz - piz;

      T inorm =
          T(6.667408e-11) / std::sqrt(dx * dx + dy * dy + dz * dz + epsilon);

      T fi = particles.m[j] * inorm * inorm * inorm;
      T fj = pim * inorm * inorm * inorm;

      ax += dx * fi;
      ay += dy * fi;
      az += dz * fi;

      particles.vx[j] -= dx * fj;
      particles.vy[j] -= dy * fj;
      particles.vz[j] -= dz * fj;
    }

    T pivx = particles.vx[i];
    T pivy = particles.vy[i];
    T pivz = particles.vz[i];

    pivx += ax;
    pivy += ay;
    pivz += az;

    pix += pivx;
    piy += pivy;
    piz += pivz;

    particles.vx[i] = pivx;
    particles.vy[i] = pivy;
    particles.vz[i] = pivz;

    particles.x[i] = pix;
    particles.y[i] = piy;
    particles.z[i] = piz;
  }
}

template <typename T> void nbody_step(particles_t<T> &particles) {
  nsimd::pack<T> grav_con(T(6.67408e-11));
  nsimd::pack<T> epsilon(T(0.00125));

  size_t len = size_t(nsimd::len(nsimd::pack<T>()));
  for (size_t i = 0; i < particles.size(); i += len) {
    nsimd::pack<T> ax(T(0));
    nsimd::pack<T> ay(T(0));
    nsimd::pack<T> az(T(0));

    nsimd::pack<T> pix = nsimd::loadu<nsimd::pack<float> >(&particles.x[i]);
    nsimd::pack<T> piy = nsimd::loadu<nsimd::pack<float> >(&particles.y[i]);
    nsimd::pack<T> piz = nsimd::loadu<nsimd::pack<float> >(&particles.z[i]);

    nsimd::pack<T> pim = nsimd::loadu<nsimd::pack<float> >(&particles.m[i]);

    for (size_t j = i + 1; j < particles.size(); ++j) {
      nsimd::pack<T> pjx = nsimd::loadu<nsimd::pack<float> >(&particles.x[j]);
      nsimd::pack<T> pjy = nsimd::loadu<nsimd::pack<float> >(&particles.y[j]);
      nsimd::pack<T> pjz = nsimd::loadu<nsimd::pack<float> >(&particles.z[j]);

      nsimd::pack<T> dx = pjx - pix;
      nsimd::pack<T> dy = pjy - piy;
      nsimd::pack<T> dz = pjz - piz;

      nsimd::pack<T> inorm =
          grav_con / nsimd::sqrt(dx * dx + dy * dy + dz * dz + epsilon);
      nsimd::pack<T> inorm3 = inorm * inorm * inorm;

      nsimd::pack<T> fi =
          nsimd::loadu<nsimd::pack<float> >(&particles.m[j]) * inorm3;
      nsimd::pack<T> fj = pim * inorm3;

      ax = ax + dx * fi;
      ay = ay + dy * fi;
      az = az + dz * fi;

      nsimd::pack<T> pjvx =
          nsimd::loadu<nsimd::pack<float> >(&particles.vx[j]);
      pjvx = pjvx - dx * fj;
      nsimd::storeu(&particles.vx[j], pjvx);
      nsimd::pack<T> pjvy =
          nsimd::loadu<nsimd::pack<float> >(&particles.vy[j]);
      pjvy = pjvy - dy * fj;
      nsimd::storeu(&particles.vy[j], pjvy);
      nsimd::pack<T> pjvz =
          nsimd::loadu<nsimd::pack<float> >(&particles.vz[j]);
      pjvz = pjvz - dz * fj;
      nsimd::storeu(&particles.vz[j], pjvz);
    }

    nsimd::pack<T> pivx = nsimd::loadu<nsimd::pack<float> >(&particles.vx[i]);
    nsimd::pack<T> pivy = nsimd::loadu<nsimd::pack<float> >(&particles.vy[i]);
    nsimd::pack<T> pivz = nsimd::loadu<nsimd::pack<float> >(&particles.vz[i]);

    pivx = pivx + ax;
    pivy = pivy + ay;
    pivz = pivz + az;

    pix = pix + pivx;
    piy = piy + pivy;
    piz = piz + pivz;

    nsimd::storeu(&particles.vx[i], pivx);
    nsimd::storeu(&particles.vy[i], pivy);
    nsimd::storeu(&particles.vz[i], pivz);

    nsimd::storeu(&particles.x[i], pix);
    nsimd::storeu(&particles.y[i], piy);
    nsimd::storeu(&particles.z[i], piz);
  }
}

int main() {

  size_t len = size_t(nsimd::len(nsimd::pack<float>()));

  size_t N = len * 4;

  particles_t<float> particles(N);

  // Sequential
  nbody_step_scalar(particles);

  // nsimd
  nbody_step(particles);

  return 0;
}
