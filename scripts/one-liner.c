/*

Copyright (c) 2020 Agenium Scale

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

/* ------------------------------------------------------------------------- */

/*

This program needs to be as portable as possible as it is intended for
Windows hosts with an unknown version of Visual Studio. It is compiled
before running the tests of NSIMD.

Its purpose is to read stdin and put all into an accumulator file and from
time to time (approximatively every second) put a line of text into another
file.

*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <errno.h>
#include <string.h>

#define DO(cmd, error_code, goto_label_on_error)                              \
  do {                                                                        \
    errno = 0;                                                                \
    if ((cmd) == error_code) {                                                \
      fprintf(stderr, "%s: error: " #cmd ": %s\n", argv[0], strerror(errno)); \
      ret = -1;                                                               \
      goto goto_label_on_error;                                               \
    }                                                                         \
  } while (0)

int main(int argc, char **argv) {
  FILE *acc, *one = NULL;
  char *buf;
  int ret = 0;
  size_t n = 1024;
  time_t tick;

  if (argc != 3) {
    fprintf(stderr, "%s: ERROR: usage: one-liner acc.txt one-liner.txt",
            argv[0]);
    return -1;
  }

  DO(acc = fopen(argv[1], "wb"), NULL, end);
  DO(buf = malloc(n), NULL, free_acc);

  tick = time(NULL);
  for (;;) {
    time_t t;
    size_t i = 0;
    int end_of_file = 0;

    for (;;) {
      int code = fgetc(stdin);
      if (code == EOF || code == '\n') {
        buf[i] = '\n';
        buf[i + 1] = 0;
        end_of_file = (code == EOF);
        break;
      }
      buf[i] = (char)code;
      if (i >= n - 2) {
        n = n * 2;
        DO(buf = realloc(buf, n), NULL, free_buf);
      }
      i++;
    }

    DO(fputs(buf, acc), EOF, free_buf);
    DO(fflush(acc), EOF, free_buf);
    t = time(NULL);
    if (t - tick >= 1) {
      DO(one = fopen(argv[2], "wb"), NULL, free_buf);
      DO(fputs(buf, one), EOF, free_one);
      DO(fflush(one), EOF, free_one);
      DO(fclose(one), EOF, free_one);
      one = NULL;
      tick = t;
    }

    if (end_of_file) {
      break;
    }
  }

  DO(one = fopen(argv[2], "wb"), NULL, free_buf);
  DO(fputs("Finished", one), EOF, free_one);
  DO(fflush(one), EOF, free_one);

free_one:
  if (one != NULL && fclose(one) == EOF) {
    fprintf(stderr, "%s: NOTE: error on closing '%s': %s\n", argv[0], argv[2],
            strerror(errno));
  }

free_buf:
  free(buf);

free_acc:
  if (fclose(acc) == EOF) {
    fprintf(stderr, "%s: NOTE: error on closing '%s': %s\n", argv[0], argv[1],
            strerror(errno));
  }

end:
  return ret;
}
