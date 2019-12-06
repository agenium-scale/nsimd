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

struct color_t {
  uint8_t r;
  uint8_t g;
  uint8_t b;

  color_t() : r(0), g(0), b(0) {}

  color_t(uint8_t const r, uint8_t const g, uint8_t const b)
      : r(r), g(g), b(b) {}
};

struct img_t {
  size_t nb_cols;
  size_t nb_rows;
  std::vector<color_t> data;

  img_t() : nb_cols(0), nb_rows(0), data() {}

  img_t(size_t const nb_cols, size_t const nb_rows)
      : nb_cols(nb_cols), nb_rows(nb_rows), data(nb_cols * nb_rows) {}

  color_t &operator()(size_t const x, size_t const y) {
    return data[y * nb_cols + x];
  }

  color_t const &operator()(size_t const x, size_t const y) const {
    return data[y * nb_cols + x];
  }
};

void write_ppm(std::string const &filename, img_t const &img) {
  std::ofstream file(filename, std::ios::binary);
  file << "P6\n";
  file << img.nb_cols << " " << img.nb_rows << "\n";
  file << "255\n";

  for (size_t i = 0; i < img.data.size(); ++i) {
    file.put(reinterpret_cast<char const &>(img.data[i].r));
    file.put(reinterpret_cast<char const &>(img.data[i].g));
    file.put(reinterpret_cast<char const &>(img.data[i].b));
  }
}

img_t mandelbrot_set(double const zoom, size_t const i_max, bool const color) {
  double const x1 = -2.1;
  double const x2 = 0.6;
  double const y1 = -1.2;
  double const y2 = 1.2;

  size_t const size_x = size_t((x2 - x1) * zoom);
  size_t const size_y = size_t((y2 - y1) * zoom);

  img_t img(size_x, size_y);

  for (size_t x = 0; x < size_x; ++x) {
    double const c_r = double(x) / zoom + x1;
    for (size_t y = 0; y < size_y; ++y) {
      double const c_i = double(y) / zoom + y1;

      double z_r = 0.0;
      double z_i = 0.0;

      size_t i = 0;

      do {
        double tmp = z_r;
        z_r = z_r * z_r - z_i * z_i + c_r;
        z_i = 2 * z_i * tmp + c_i;
        ++i;
      } while (z_r * z_r + z_i * z_i < 4 && i < i_max);

      if (color) {
        if (i != i_max) {
          int r = 0;
          int g = 0;
          int b = int(i * 255 * 42 / i_max);
          if (b > 255) {
            g = b - 255;
            b = 255;
          }
          if (g > 255) {
            r = g - 255;
            g = 255;
          }
          if (r > 255) {
            r = 255;
          }
          img(x, y) = color_t(uint8_t(r), uint8_t(g), uint8_t(b));
        }
      } else {
        if (i == i_max) {
          img(x, y) = color_t(255, 255, 255);
        }
      }
    }
  }

  return img;
}

img_t mandelbrot_set_nsimd(double const zoom, size_t const i_max,
                           bool const color) {
  double const x1 = -2.1;
  double const x2 = 0.6;
  double const y1 = -1.2;
  double const y2 = 1.2;

  size_t const size_x = size_t((x2 - x1) * zoom);
  size_t const size_y = size_t((y2 - y1) * zoom);

  img_t img(size_x, size_y);

  size_t len = size_t(nsimd::len(nsimd::pack<double>()));

  for (size_t x = 0; x < size_x; ++x) {
    nsimd::pack<double> const c_r = double(x) / zoom + x1;
    for (size_t y = 0; y + len < size_y; y += len) {
      nsimd::pack<double> const c_i = double(y) / zoom + y1; // TODO

      nsimd::pack<double> z_r = 0.0;
      nsimd::pack<double> z_i = 0.0;

      double i = 0;

      nsimd::pack<double> i_pack = -1.0;
      nsimd::packl<double> cond;

      do {
        nsimd::pack<double> tmp = z_r;
        z_r = z_r * z_r - z_i * z_i + c_r;
        z_i = 2 * z_i * tmp + c_i;
        ++i;

        cond = z_r * z_r + z_i * z_i < 4;
        i_pack = nsimd::if_else1(cond, nsimd::pack<double>(i), i_pack);
      } while (nsimd::any(cond) && i < i_max);

      if (color) {
        if (i != i_max) {
          int r = 0;
          int g = 0;
          int b = int(i * 255 * 42 / i_max);
          if (b > 255) {
            g = b - 255;
            b = 255;
          }
          if (g > 255) {
            r = g - 255;
            g = 255;
          }
          if (r > 255) {
            r = 255;
          }
          img(x, y) = color_t(uint8_t(b), uint8_t(g), uint8_t(r));
        }
      } else {
        if (i == i_max) {
          img(x, y) = color_t(255, 255, 255);
        }
      }
    }
  }

  return img;
}

int main() {
  float const zoom = 500.0;
  size_t const i_max = 20;

  std::cout << "mandelbrot_set(zoom = " << zoom << ", i_max = " << i_max << ")"
            << std::endl;
  img_t const img_seq = mandelbrot_set(zoom, i_max, true);
  write_ppm("build/mandelbrot_seq.ppm", img_seq);

  std::cout << "mandelbrot_set_nsimd(zoom = " << zoom << ", i_max = " << i_max
            << ")" << std::endl;
  img_t const img_nsimd = mandelbrot_set_nsimd(zoom, i_max, true);
  write_ppm("build/mandelbrot_nsimd.ppm", img_nsimd);

  return 0;
}
