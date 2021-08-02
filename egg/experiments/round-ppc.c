#include <altivec.h>
#include <stdio.h>

void pp(const char *prefix, FILE *out, float buf[4]) {
  fputs(prefix, out);
  fputc('{', out);
  for (int i = 0; i < 4; i++) {
    fprintf(out, " %f", (double)buf[i]);
  }
  fputs(" }\n", out);
}

int main() {
  float res[4];

  float buf[4];
  buf[0] = -1.5f;
  buf[1] = -0.5f;
  buf[2] = 0.5f;
  buf[3] = 1.5f;
  __vector float v = *(__vector float *)buf;


  pp("   buf = ", stdout, buf);


  *(__vector float *)res = vec_round(v);
  pp(" round = ", stdout, res);

  *(__vector float *)res = vec_rint(v);
  pp("  rint = ", stdout, res);

  *(__vector float *)res = vec_roundc(v);
  pp("roundc = ", stdout, res);

  return 0;
}
