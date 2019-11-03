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

template <typename A>
void read_pgm(std::string const &filename, std::vector<uint8_t, A> &img,
              size_t &nb_cols, size_t &nb_rows) {
  std::ifstream file(filename, std::ios::binary);
  if (file.good() == false) {
    throw std::runtime_error("Cannot open file \"" + filename + "\"");
  }

  std::string magic_number;
  std::getline(file, magic_number);
  if (magic_number != "P5") {
    throw std::runtime_error("Unsupported magic number \"" + magic_number +
                             "\" (only P5 is supported)");
  }
  if (file.good() == false) {
    throw std::runtime_error("Cannot read file \"" + filename +
                             "\" after the magic number \"" + magic_number +
                             "\"");
  }

  std::string nb_cols_str;
  std::string nb_rows_str;
  file >> nb_cols_str >> nb_rows_str;
  {
    std::string tmp;
    std::getline(file, tmp);
  }
  if (file.good() == false) {
    throw std::runtime_error("Cannot read file \"" + filename +
                             "\" after the size \"" + nb_cols_str + " " +
                             nb_rows_str + "\"");
  }

  std::string max_value;
  std::getline(file, max_value);
  if (max_value != "255") {
    throw std::runtime_error("Unsupported max value \"" + max_value +
                             "\" (only 255 is supported)");
  }
  if (file.good() == false) {
    throw std::runtime_error("Cannot read file \"" + filename +
                             "\" after the max value \"" + max_value + "\"");
  }

  nb_cols = size_t(std::stoi(nb_cols_str));
  nb_rows = size_t(std::stoi(nb_rows_str));

  img.reserve(nb_rows * nb_cols);

  while (file.good()) {
    uint8_t c;
    file.get(reinterpret_cast<char &>(c));
    if (img.size() == nb_cols * nb_rows && c != uint8_t('\n')) {
      // Ignore final \n
    } else {
      img.push_back(c);
    }
  }

  if (img.size() != nb_cols * nb_rows) {
    throw std::runtime_error("The size of the image \"" + filename + "\" is " +
                             std::to_string(img.size()) + " instead of " +
                             std::to_string(nb_cols) + "x" +
                             std::to_string(nb_rows) + " (" +
                             std::to_string(nb_cols * nb_rows) + ")");
  }
}

template <typename A>
void write_pgm(std::string const &filename, std::vector<uint8_t, A> const &img,
               size_t const nb_cols, size_t const nb_rows) {
  std::ofstream file(filename, std::ios::binary);
  file << "P5\n";
  file << nb_cols << " " << nb_rows << "\n";
  file << "255\n";

  for (size_t i = 0; i < img.size(); ++i) {
    file.put(reinterpret_cast<char const &>(img[i]));
  }
}

template <typename A>
void threshold(std::vector<uint8_t, A> &img, uint8_t const &threshold_value) {
  for (size_t i = 0; i < img.size(); ++i) {
    if (img[i] < threshold_value) {
      img[i] = 0;
    } else {
      img[i] = 255;
    }
  }
}

template <typename A>
void threshold_nsimd(std::vector<uint8_t, A> &img,
                     uint8_t const &threshold_value) {
  nsimd::pack<uint8_t> pack_threshold_value(threshold_value);
  nsimd::pack<uint8_t> pack_0(0);
  nsimd::pack<uint8_t> pack_255(255);

  size_t i;
  size_t len = size_t(nsimd::len(nsimd::pack<uint8_t>()));
  for (i = 0; i + len < img.size(); i += len) {
    nsimd::pack<uint8_t> pack = nsimd::loada<nsimd::pack<uint8_t> >(&img[i]);
    nsimd::packl<uint8_t> pack_logical = pack < pack_threshold_value;
    // Possible values:
    // { 1, 0, 0, 1 } = { 75, 180, 152, 40, } < { 128, 128, 128, 128 }
    pack = nsimd::if_else1(pack_logical, pack_0, pack_255);
    // Possible values:
    // { 255, 0, 0, 255 } = if_else1({ 0, 0, 0, 0, }, { 255, 255, 255, 255 })
    nsimd::storea(&img[i], pack);
  }

  for (; i < img.size(); ++i) {
    if (img[i] < threshold_value) {
      img[i] = 0;
    } else {
      img[i] = 255;
    }
  }
}

int main() {

  std::string const img_filename =
      "img/640px-Pavlovsk_Railing_of_bridge_Yellow_palace_Winter.pgm";

  std::vector<uint8_t, nsimd::allocator<uint8_t> > img;
  size_t nb_cols;
  size_t nb_rows;
  read_pgm(img_filename, img, nb_cols, nb_rows);
  std::cout << "Load image \"" << img_filename << "\", size " << nb_cols << "x"
            << nb_rows << " (img size = " << img.size() << ")" << std::endl;

  std::vector<uint8_t, nsimd::allocator<uint8_t> > img_cpu = img;
  threshold(img_cpu, uint8_t(128));
  write_pgm("build/threshold_cpu.pgm", img_cpu, nb_cols, nb_rows);

  std::vector<uint8_t, nsimd::allocator<uint8_t> > img_nsimd = img;
  threshold_nsimd(img_nsimd, uint8_t(128));
  write_pgm("build/threshold_nsimd.pgm", img_nsimd, nb_cols, nb_rows);

  return 0;
}
