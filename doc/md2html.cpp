/*

Copyright (c) 2019 Agenium Scale

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#include <stdexcept>
#include <ns2.hpp>

// ----------------------------------------------------------------------------

// Extract lines form strings like ":L7:L42"
// Returns -1 if fails
std::pair<int, int> extract_lines(std::string const &s) {
  std::pair<int, int> r(-1, -1);
  std::vector<std::string> lines = ns2::split(s, ":L");
  if (lines.size() == 3 && lines[0] == "") {
    try {
      r.first = std::stoi(lines[1]);
      r.second = std::stoi(lines[2]);
    } catch (std::exception const &e) {
      // r.first and r.second are -1
    }
  }
  return r;
}

// ----------------------------------------------------------------------------

std::string callback_input_filename = "";

std::string callback_macro(std::string const &label, std::string const &url,
                           ns2::markdown_infos_t const &markdown_infos) {
  std::string filename;
  if (ns2::startswith(label, "INCLUDE")) {
    filename = ns2::join_path(ns2::dirname(callback_input_filename), url);
  }

  std::string lang;
  if (ns2::startswith(label, "INCLUDE_CODE")) {
    std::string const ext = ns2::splitext(filename).second;
    if (ext == "sh") {
      lang = "Bash";
    } else if (ext == "c" || ext == "h") {
      lang = "C";
    } else if (ext == "cpp" || ext == "hpp") {
      lang = "C++";
    } else if (ext == "py") {
      lang = "Python";
    }
  }

  if (ns2::startswith(label, "INCLUDE_CODE:")) {
    std::string const lines_str = label.substr(label.find(':'));
    std::pair<int, int> const l_first_last = extract_lines(lines_str);
    if (l_first_last.first == -1) {
      throw std::runtime_error("cannot extract first line number");
    }
    if (l_first_last.second == -1) {
      throw std::runtime_error("cannot extract last line number");
    }
    std::string out;
    std::string lines;
    {
      ns2::ifile_t in(filename);
      int num_line = 1;
      std::string line;
      while (std::getline(in, line)) {
        if (num_line == l_first_last.second) {
          lines += line;
        } else if (num_line < l_first_last.second) {
          if (num_line >= l_first_last.first) {
            lines += line + "\n";
          }
        } else {
          break;
        }
        ++num_line;
      }
    }
    ns2::compile_markdown("```" + lang + "\n" + ns2::deindent(lines) +
                              "\n```\n",
                          &out, markdown_infos);
    return out;
  }

  if (ns2::startswith(label, "INCLUDE_CODE")) {
    std::string out;
    ns2::compile_markdown("```" + lang + "\n" + ns2::read_file(filename) +
                              "\n```\n",
                          &out, markdown_infos);
    return out;
  }

  if (ns2::startswith(label, "INCLUDE")) {
    ns2::ifile_t in(filename);
    std::ostringstream out;
    ns2::compile_markdown(&in, &out, markdown_infos);
    return out.str();
  }

  return "";
}

// ----------------------------------------------------------------------------

std::pair<std::string, bool>
callback_link(std::string const &label, std::string const &url,
              ns2::markdown_infos_t const &markdown_infos) {
  if (markdown_infos.output_format != ns2::HTML) {
    return std::pair<std::string, bool>("", false);
  }

  std::pair<std::string, std::string> root_basename_ext = ns2::splitext(url);
  if (root_basename_ext.second == "md") {
    return std::pair<std::string, bool>(
        ns2::html_href(root_basename_ext.first + ".html", label), true);
  } else {
    return std::pair<std::string, bool>("", false);
  }
}

// ----------------------------------------------------------------------------

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <input_file> <output_file>"
              << std::endl;
    return 1;
  }

  std::string const input_filename = argv[1];
  std::string const output_filename = argv[2];

  ns2::ifile_t input_file(input_filename);
  ns2::ofile_t output_file(output_filename);

  std::cout << "Convert \"" << input_filename << "\" to \"" << output_filename
            << "\"" << std::endl;

  callback_input_filename = input_filename;
  ns2::markdown_infos_t markdown_infos(ns2::HTML, callback_macro,
                                       callback_link, true);

  ns2::compile_markdown(&input_file, &output_file, markdown_infos);

  return 0;
}
