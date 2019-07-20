#include <stdio.h>
#include <arm_sve.h>

// armclang -march=armv8+sve egg/experiments/upcvt-sve.c -o ../build/a.out

// ---

int len32() {
  return (int)svcntp_b32(svptrue_b32(), svptrue_b32());
}

void print32(FILE *out, const char *var, svfloat32_t a) {
  float buf[2048];
  svst1_f32(svptrue_b32(), buf, a);
  fprintf(out, "%s = ", var);
  for (int i = 0; i < len32(); i++) {
    if (i > 0) {
      fputs(", ", out);
    }
    fprintf(out, "%f", (double)buf[i]);
  }
  fputc('\n', stdout);
}

svfloat32_t iota32(float i0) {
  float buf[2048];
  for (int i = 0; i < len32(); i++) {
    buf[i] = i0 + (float)i;
  }
  return svld1(svptrue_b32(), buf);
}

// ---

int len64() {
  return (int)svcntp_b64(svptrue_b64(), svptrue_b64());
}

void print64(FILE *out, const char *var, svfloat64_t a) {
  double buf[2048];
  svst1_f64(svptrue_b64(), buf, a);
  fprintf(out, "%s = ", var);
  for (int i = 0; i < len64(); i++) {
    if (i > 0) {
      fputs(", ", out);
    }
    fprintf(out, "%f", buf[i]);
  }
  fputc('\n', stdout);
}


// ---

int main() {
  svfloat32_t a = iota32(0.0f);
  svfloat32_t b = iota32(8.0f);
  svfloat64_t c = svcvt_f64_f32_z(svptrue_b32(), svzip1_f32(a, a));
  print32(stdout, "a ", a);
  print32(stdout, "aa", svzip1_f32(a, a));
  print64(stdout, "c ", c);
  return 0;
}
