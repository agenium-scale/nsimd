<!--

Copyright (c) 2021 Agenium Scale

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

-->

# How tests are done?

First and foremost note that this a work in progress and that we are doing our
best to have serious testing of the library.

We can also state our conclusion on testing: we are not and never will be
satisfied with our tests, there are not enough of them, we want more.

The current system has on average 10000 tests by SIMD extensions. Thanks to
our "Python" approach we can automatically generate tests for all operators
and for all types. This has greatly helped us in finding bugs. But, as you
know, bugs are always there.

## Why write this?

Testing the library has been taken seriously since its very beginning. Tests
have gone through several stages:

- The first one was during the development of the first version of the library.
  Tests of operators were done with random numbers as input. Those random
  numbers were all powers of 2 to ease the comparisons of basic arithmetic
  types. NaNs and infinities were not generated as inputs and operators
  behaviors with those inputs were not tested

- For the second stage random numbers generators have been improved to emit
  NaNs and infinities. It allowed us to detect many errors in operators,
  mostly in math functions like cos, sin, exp... But we also discovered bugs
  in hardware when NaNs and infinities are given to intrinsics.

- The third stage which the current test system takes into account the
  experience we gain with the privous two. As we have abandonned the buggy and
  slow implementations of math functions coming from Boost.SIMD and now rely on
  the excellent Sleef (<https://sleef.org/>) we trust that the math functions
  are correctly tested. In more details we do not generate NaNs and infinities
  anymore because we trust functions coming from Sleef and we do not want
  to write code in our tests to bypass hardware bugs. We only care that our
  wrapping are correct adn that `nsimd::add` correctly calls add, the fact that
  the add does not work correctly is a hardware bug then and not the
  problem of the library.

Tests on floatting points are done using ULPs. ULP means units in the last
place and is commonly used for the comparison of floatting point numbers.
It is in general a bad idea to compare floats with the `==` operators as
it essentially compares bits. Instead we want to check if the results of
two computations are "not to far away from each other". When checking an
operator, let's say, on CPUs and GPUs, we to take into account that
- the rounding mode may be different and
- the precision of the calculation may be different.

## ULPs

This chapter is dedicated to math proof concerning ULPs. Indeed people use
this notion but proofs are hard to find. We give our own definition of distance
in ULPs, compare it to the usual one and give pros and cons.
We assume the reader is familiar with basic mathematics.

For this entire chapter fix the following:
- an integer $b > 1$ (will be our radix),
- an integer $p > 1$ (will be the number of digits in the mantissa)
- an integer $M > 1$ (will be the minimum exponent allowed for floatting
  point numbers)
A floatting point number is an element of $\mathbb{R}$ of the form
$m b^e$ with $e \geq -M$ and $m \in \mathbb{Z}$. More precisely we define
the set of floatting point numbers $F$ to be the union of the following two
sets:
- $\{ mb^e \in F \text{ with } e > -M \}$ the *normal* numbers.
- $\{ mb^{-M} \in F \text{ with } m \in \mathbb{Z} \text{ and }
  0 < |m| < b^p \}$ the *denormal* or *subnormal* numbers.

The set $F$ can be viewed as a subset of $\mathbb{R}$ with the mapping
$\phi : (m, e) \mapsto mb^e$ and we will make this abuse of
notation in what follows. Usually the sign of the floatting point number
is separated from $m$ but we include it "inside" $m$ as it does not change
the proofs below and simplifies the notations.

Let $a_i \in F$ for $i = 1,2$ such that $a_i = m_i b^{e_i}$.

**Proposition:** $\phi$ is injective.

**Proof:** Suppose that $a_1 = a_2$ or $m_1b^{e_1} = m_2b^{e_2}$. If $a_1$
and $a_2$ are subnormal numbers then $e_1 = e_2 = -M$ and $m_1 = m_2$. If
$a_1$ and $a_2$ are normal numbers suppose that $e_2 > e_1$, then
$|\frac{m_2b^{e_2}}{m_1b^{e_1}}| > b^{e_2 + p - 1 - e_1 - p}
= b^{e_2 - e_1 - 1} \geq b^{1 - 1} = 1$ therefore
$m_2b^{e_2} \neq m_1b^{e_1}$ which is absurd hence $e_1 = e_2$ and as a
consequence $m_1 = m_2$.

**Definition:** We define the *distance in ULPs between $a_1$ and $a_2$*
denoted by $U(a_1, a_2)$ to be:
- $|m_1b^{e_1 - e_2} - m_2|$ if $e_1 \geq e_2$,
- $|m_1 - m_2b^{e_2 - e_1}|$ otherwise.

**Example:** Take $a_1 = 123456 \times 10^5$ and $a_2 = 123789 \times 10^5$
Then as the exponents of $a_1$ and $a_2$ are the same we have
$U(123456 \times 10^5, 123789 \times 10^5) = |123789 - 123456| = 333$.

The following proposition confort the name "units in the last place".

**Proposition:** Let $f = \lfloor \log_b U(a_1, a_2) \rfloor + 1$ and suppose
that $a_1, a_2$ are of same sign and have the same exponents, then either the
first $p - f$ digits of $m_1$ and $m_2$ are identical or their difference is
$\pm 1$.

**Proof:** For $i = 1,2$ there exists $q_i \in \mathbb{Z}$ and
$0 \leq r_i < b^f$ such that $m_i = q_i b^f + r_i$. Then
$|q_1 - q_2| \leq \frac{|m_1 - m_2| + |r_1 - r_2|}{b^f}
< \frac{b^{\log_b(U(a_1, a_2)} + b^f}{b^f} = 2$

So that either $q_1 = q_2$ or $q_1 - q_2 = \pm 1$. It is interesting to know
what are the cases when $q_1 - q_2 \pm 1$. Suppose that $0 \leq m_1 < m_2$
and that $q_1 = q_2 + 1$ then $m_1 = q_1 b^f  + r_1 \geq q_2 b^f + b^f >
q_2 b^f + r_2 = m_2$ which contradicts the hypothesis hence $q_1 \leq q_2$.
Finally $r_1 + U(a_1, a_2) = r_1 + (m_2 - m_1) = q_2 b^f + r_2 - q_1 b^f
= r_2 + b_f$ so that:
- $r_1 + U(a_1, a_2) \geq b^f$ and
- $r_1 = r_2 + (b_f - U(a_1, a_2)) = r_2 + (b^f - b^{\log_b(U(a_1, a_2))})
  > r_2$.

**Example:** Taking back $a_1 = 123456 \times 10^5$ and
$a_2 = 123789 \times 10^5$. As $q_1 = q_2$ we have the first 3 digits of $a_1$
and $a_2$ that are identical and they differ by their last
$\log_{10} \lfloor U(a_1, a_2) \rfloor + 1
= \lfloor \log_{10}(333) \rfloor + 1 = 3$

**Example:** Now take $a_1 = 899900 \times 10^5$ and
$a_2 = 900100 \times 10^5$. We have $f = 3$ but $q_2 = q_1 + 1$ and
$r_2 = 900 > 100 = r_1$ and $r_2 + U(a_1, a_2) = 1100 \geq 1000 = 10^3$.

The propositions above show that our definition of the ULP distance is well
choosen as we have the following results:
- (second proposition) is measures de number of different digits at the end
  of the mantissa.
- (first proposition) if we write the numbers differently but still in base $b$
  we only change the number of different digits in the last places by some
  zeros. The latter number being the exponent of $b$ that represents the
  difference in scaling of both representations of floatting point numbers.

We show now how to compute it using the IEEE 754 floatting point numbers
representation. A floatting point number $(m, e) \in F$ is stored in memory
(and registers) as the integer $\pm ((e + M)b^p + |m|)$.

**Proposition:** If $e_2 \geq e_1 + 2$ then $U(a_1, a_2) \geq b^p$.

**Proof:** We have $U(a_1, a_2) = |m_2 b^{e_2 - e_1} - m_1|
\geq ||m_2| b^{e_2 - e_1} - |m_1||$. But $m_2$ is a normal number otherwise we
would have $e_2 = -M = e_1$ so that $|m_2| \geq b^{p - 1}$ and we have
$|m_2| b^{e_2 - e_1} \geq b^{p - 1 + e_2 - e_1} \geq b^{p + 1} > |m_1|$,
therefore $||m_2| b^{e_2 - e_1} - |m_1|| \geq |m_2|b^2 - |m_1|
> b^{p - 1 + 2} - b^p = b^p$.

The proposition above basically states that if two floatting point numbers
are two orders of magnitude away then that have no digits in common, and
that there are godd chances that comparing them is not interesting at all.

The usual definition of the distance in ULPs is roughly given as the number
of floatting point numbers between the two considered floatting point numbers.
More precisely we will denote it by $V$ and it is defined as follows:
- $V(a_1, a_2) = |(e_1 + M)b^p + |m_1| - (e_2 + M)b^p - |m_2||$ if $a_1$ and
  $a_2$ have the same signs
- $V(a_1, a_2) = (e_1 + M)b^p + |m_1| + (e_2 + M)b^p + |m_2|$ otherwise.

**Proposition:** If $e_1 = e_2$ and $a_1$, $a_2$ have the same sign then
$U(a_1, a_2) = V(a_1, a_2)$.

**Proof:** We have $V(a_1, a_2) = |(e_1 + M)b^p + m_1 - (e_2 + M)b^p - m_2|$,
but as $e_1 = e_2$, we end up with $V(a_1, a_2) = |m_1 - m_2| = U(a_1, a_2)$.

**Proposition:** $V(a_1, a_2) = 1$ is equivalent to $U(a_1, a_2) = 1$.

**Proof:** The proposition is true if $e_1 = e_2$. Suppose that $e_2 > e_1$.
Note that $a_2$ is a normal number so that $m_2 \geq b^{p - 1}$.

We first suppose that $V(a_1, a_2) = 1$. Then by the definition of $V$, $a_1$
and $a_2$ have same sign otherwise $V(a_1, a_2) \geq 2$ and we suppose that
$a_i \geq 0$. Moreover we have $e_2 = e_1 + 1$ otherwise we would have that
$a_1 = m_1b^{e_1} < m_1b^{e_1 + 1} < m_2b^{e_1 + 2} \leq a_2$. Now we have
$(b^p - 1)b^{e_1} < b^{p - 1}b^{e_1 + 1}$ and let
$(b^p - 1)b^{e_1} \leq mb^e \leq b^{p - 1}b^{e_1 + 1}$.

First note that if $a = mb^e$ is a normal number then $m \geq b^{p - 1}$ and if
$a$ is a subnormal number then $e = -M$ in which case we also have $e_1 = -M$
and $m \geq b^p - 1 \geq b^{p - 1}$. In any case $m \geq b^{p - 1}$.

We have $(b^p - 1)/m b^{e_1} < b^e < b^{p - 1}/m b^{e_1 + 1}$. But
$1 \leq (b^p - 1) / m$ and $b^{p - 1} / m \leq 1$ so that
$b^{e_1} \leq b^e \leq b^{e_1 + 1}$ and $e = e_1$ or $e = e_1 + 1$. In the
first case $(b^p - 1)b^{e_1} \leq mb^{e_1}$ so that $b^p - 1 \leq m$ but
$m < b^p$ and $m = b^p - 1$. In the second case
$mb^{e_1 + 1} \leq b^{p - 1}b^{e_1 + 1}$ so that $m \leq b^{p - 1}$ but
$b^{p - 1} \leq m$ and $m = b^{p - 1}$. We have proven that two consecutive
elements of $F$ with $e_2 = e_1 + 1$ are neessary of the form
$a_1 = (b^p - 1)b^{e_1}$ and $a_2 = b^{p - 1}b^{e_1 + 1}$. Now we can compute
$U(a_1, a_2) = |bb^{p - 1} - (b^p - 1)| = 1$.

Conversely, suppose that $U(a_1, a_2) = 1$, then
$|b^{e_2 - e_1}m_2 - m_1| = 1$. Suppose that $b^{e_2 - e_1}m_2 - m_1 = -1$,
then $-1 \geq bb^{p - 1} - b^p = 0$ which is absurd. We then have
$b^{e_2 - e_1}m_2 - m_1 = 1$. Suppose that $e_2 \geq e_1 + 2$ then we would
have that $b^{e_2 - e_1}m_2 - m_1 \geq b^2b^{p - 1} - b^p \geq b^p$ which is
absurd so that $e_2 = e_1 + 1$ and $bm_2 - m_1 = 1$. Suppose that
$m_2 \geq b^{p - 1} + 1$ then $bm_2 - m_1 \geq b^p + b - (b^p - 1) \geq 2$
which is absurd so that $m_2 = b^{p - 1}$ and as a consequence $m_1 = b^p - 1$.

If $a_1, a_2 < 0$, then $V(a_1, a_2) = 1$ is equivalent by definition to
$V(-a_1, -a_2) = 1$ which is equivalent to $U(-a_1, -a_2) = 1$ which is
by definition equivalent to $U(a_1, a_2) = 1$.

**Proposition:** Suppose that $e_1 \leq e_2 \leq e_1 + 1$ then
$V \leq U \leq bV$.

**Proof:** The proposition is true if $e_1 = e_2$. Suppose now that
$e_2 = e_1 + 1$. Then we have
$b^p + m_2 - m_1 \geq b^p + b^{p - 1} - b^p \geq 0$
so that $V(a_1, a_2) = b^p + m_2 - m_1 = b^p + m_2(1 - b) + bm_2 - m_1$. But
$b^p + m_2(1 - b) \leq b^p + b^p(1 - b) \leq 0$ and
$bm_2 - m_1 \geq bb^{p - 1} - b^p = 0$ so that $V(a_1, a_2) \leq bm_2 - m_1
= U(a_1, a_2)$. On the other hand we have $bm_2 - m_1
\leq b(b^p + m_2 - m_1 + m_1 - m_1/b - b^p)$ but
$m_1 - m_1/b - b^p \leq b^p - b^{p - 1}/b - b^p \leq 0$ so that
$U(a_1, a_2) \leq b(b^p + m_2 - m_1) = bV(a_1, a_2)$.

**Remark:** The previous propositions shows that the difference between $V$
and $U$ is only visible when the arguments have differents exponents and
are non consecutive. Our version of the distance in ULPs puts more weights
when crossing powers of $b$. Also if $e_2 \geq e_1 + 2$ then we have seen that
$a_1$ and $a_2$ have nothing in common which is indicated by the fact that
$U, V \geq b^p$.

**Definition:** We now define the relative distance $D(a_1, a_2)$ between
$a_1$ and $a_2$ to be $|a_1 - a_2| / \min(|a_1|, |a_2|)$.

**Proposition:** As $U$ is defined in a "mathematical" way compared to $V$ then
the relation between $U$ and $D$ is straightforward and we have
$D(a_1, a_2) = U(a_1, a_2) / |m_1|$. Moreover we have
$b^{-q}U \leq D \leq b^{1 - q}U$ where $q$ is the greatest integer such that
$b^{q - 1} \leq |m_1| < b^q$. In particular if $a_1$ is a normal number then
$p = q$.

**Proof:** Suppose that $|a_1| < |a_2|$, then we have three cases:
- If $a_2$ is denormal, then so is $a_1$ and $e_1 = -M = e_2$.
- If $a_2$ is normal, then:
  + If $a_1$ is denormal then $e_1 < e_2$.
  + If $a_1$ and $a_2$ are normal numbers then $|m_1/m_2| b^{e_1 - e_2} < 1$
    but $|m_1/m_2| \geq b^{p - 1} / b^p = b^{-1}$ and we have
    $b^{e_1 - e_2 - 1} < 1$ so that $e_1 < e_2 + 1$ or $e_1 \leq e_2$.
In any case we have $e_1 \leq e_2$, as a consequence we have
$D(a_1, a_2) = |m_1b^{e_1} - m_2b^{e_2}| / \min(|m_1|b^{e_1}, |m_2|b^{e_2})
= |m_1 - m_2b^{e_2 - e_1}| / \min(|m_1|, |m_2|b^{e_2 - e_1})$. Therefore
$D(a_1, a_2) = U(a_1, a_2) / \min(|m_1|, |m_2|b^{e_2 - e_1})$. Now if
$e_1 = e_2$ then $\min(|m_1|, |m_2|) = |m_1|$ but if $e_2 > e_1$ then $a_2$ is
a normal number and $|m_1| < b^p = b \times b^{p - 1} \leq b^{e_2 - e_1} |m_2|$
and again $\min(|m_1|, |m_2|b^{e_2 - e_1}) = |m_1|$.

Applying $b^{q - 1} \leq |m_1| < b^q$ we get that
$b^{-q}U \leq D \leq b^{1 - q}U$. If moreover $a_1$ is a normal number then
by definition $p = q$.

**Remark:** Using the inequality of the previous proposition and taking the
base-$b$ logarithm we get $-q + \log U \leq \log D \leq 1 - q + \log U$ and
then $-q + \lfloor \log U \rfloor \leq \lfloor \log D \rfloor
\leq 1 - q + \lfloor \log U \rfloor$ hence two possibilities:
- $-q + \lfloor \log U \rfloor = \lfloor \log D \rfloor$ in which case
  $\lfloor \log U \rfloor + (-\lfloor \log D \rfloor) = q$.
- $1 - q + \lfloor \log U \rfloor = \lfloor \log D \rfloor$ in which case
  $1 + \lfloor \log U \rfloor + (-\lfloor \log D \rfloor) = q$.
According to a above proposition we know that $f = 1 + \lfloor \log U \rfloor$
can be interpreted as the number of differents digits in the last places of the
mantissa. Write $\mathcal{D} = - \lfloor \log D \rfloor$ then
$q \leq f + \mathcal{D} \leq q + 1$. The latter inequality shows that
$\mathcal{D}$ can be interpreted as the number of digits which are the same in
the mantissa near the "first" place. Note that for denormal numbers the "first"
places are near the bit of most significance. We can conclude this remark with
the interpretation that two floatting point numbers have at least
$\mathcal{D} - 1$ digits in common in the first place of the mantissa and $f$
digits which are different in the last place of the mantissa.

**Algorithm:** We give below the C code for $U$ with a caveat. As seen in a
previous proposition when $e_2 \geq e_1 + 2$ the arguments have no digit in
common and can be considered too far away in which case we return `INT_MAX` (or
`LONG_MAX`). As a side effect is that the code will be free of multiprecision
integers (which would be necessary as soon as $|e_2 - e_1| \geq 12$) hence
lesser dependencies, readability, maintainability and performances.
When $|e_2 - e_1| \leq 1$ we use the formula of the definition.

```c
/* We suppose that floats are IEEE754 and not NaN nor infinity */

struct fl_t{
  int mantissa;
  int exponent;
};

fl_t decompose(float a_) {
  fl_t ret;
  unsigned int a;
  memcpy(&a, &a_, sizeof(float)); /* avoid aliasing */
  ret.exponent = (int)((a >> 23) & 0xff) - 127;
  if (ret.exponent == -127) {
    /* denormal number */
    ret.mantissa = (int)a;
  } else {
    ret.mantissa = (int)((1 << 23) | a);
  }
  if (a >> 31) {
    ret.mantissa = -ret.mantissa;
  }
  return ret;
}

int distance_ulps(float a_, float b_) {
  fl_t a, b;
  a = decompose(a_);
  b = decompose(b_);

  if (a.exponent - b.exponent < -1 || a.exponent - b.exponent > 1) {
    return INT_MAX;
  }
  
  int d;
  if (a.exponent == b.exponent) {
    d = a.mantissa = b.mantissa;
  } else if (a.exponent > b.exponent) {
    d = 2 * a.mantissa - b.mantissa;
  } else {
    d = 2 * b.mantissa - a.mantissa;
  }

  return d > 0 ? d : -d;
}
```

The algorithm for computing $\mathcal{D} - 1$ follows:

```c
int d(float a_, float b_) {
  float absa = fabsf(a_);
  float absb = fabsf(b_);

  /* ensure that |a_| <= |b_| */
  if (absb < absa) {
    float tmp = absa;
    absa = absb;
    absb = tmp;
  }

  fl_t a = decompose(a_);
  int q = 0;
  for (q = 0; q <= 23 && (2 << q) <= a.mantissa; q++);

  int ulps = distance_ulps(a_, b_);
  int lu;
  for (lu = 0; lu <= 30 && (2 << (lu + 1)) <= a.mantissa; lu++);

  return q - (lu + 1) - 1;
}
```
