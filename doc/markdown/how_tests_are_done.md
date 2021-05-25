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
- $F = \{ m \in \mathbb{Z} \text{ with } |m| < b^p \} \times \mathbb{Z}$
  (the set of floatting point numbers)

The set $F$ can be viwed as a subset (non injectively) of $\mathbb{R}$ with
the mapping $\phi : (m, e) \mapsto mb^e$ and we will make this abuse of
notation in what follows.

Let $a_i \in F$ for $i = 1,2$ such that $a_i = m_i b^{e_i}$ with $|m_i| < b^p$
and $e_i \in \mathbb{Z}$. We define the *distance in ULPs between $a_1$ and
$a_2$* denoted by $U(a_1, a_2)$ to be:
- $|m_1b^{e_1 - e_2} - m_2|$ if $e_1 \geq e_2$,
- $|m_1 - m_2b^{e_2 - e_1}|$ otherwise.

**Proposition:** Suppose that $\phi(a_1) = \phi(a_1')$ and
$\phi(a_2) = \phi(a_2')$. Then $U(a_1, a_2) / U(a_1', a_2')$ (or its inverse)
is a power of $b$. If moreover we impose that $|m_i| \geq b^{p - 1}$ then
$\phi$ will be injective and $U(a_1, a_2) = U(a_1', a_2')$.

**Proof:** Suppose that for $i = 1,2$, we have  $m_i b^{e_i} = m_i' b^{e_i'}$.
Suppose that $e_1 \geq e_2$ and $e_1' \geq e_2'$ then $|m_1b^{e_1 - e_2} - m_2|
= |m_1'b^{e_1' - e_2} - m_2'b^{e_2' - e_2}|
= b^{e_2' - e_2} |m_1'b^{e_1' - e_2'} - m_2'|$. Now suppose that $e_1' < e_2'$,
$|m_1b^{e_1 - e_2} - m_2| = |m_1'b^{e_1' - e_2} - m_2'b^{e_2' - e_2}|
= b^{e_1' - e_2} |m_1' - m_2'b^{e_2' - e_1'}|$.

Finally suppose that $mb^e = nb^f$ with $b^{p - 1} \leq |m|, |n| < b^p$.
Suppose that $e > f$, then $|\frac{mb^e}{nb^f}| > b^{e + p - 1 - f - p}
= b^{e - f - 1} \geq b^{1 - 1} = 1$ therefore $mb^e \neq nb^f$ which is absurd
hence $e = f$ and as a consequence $m = n$.

**Example:** Take $a_1 = 123456 \times 10^5$ and $a_2 = 123789 \times 10^5$
Then as the exponents of $a_1$ and $a_2$ are the same we have
$U(123456 \times 10^5, 123789 \times 10^5) = |123789 - 123456| = 333$.

The following proposition confort the name "units in the last place".

**Proposition:** Let $f = \lfloor \log_b U(a_1, a_2) \rfloor + 1$ and suppose
that $a_1, a_2$ are of same sign and have the same exponents, then either the
first $p - f$ digits of $m_1$ and $m_2$ are identical or their difference is
$\pm 1$.

**Proof:** For $i = 1,2$ there exists $q_i \in \mathbb{Z}$ and
$0 \leq r_i < p^f$ such that $m_i = q_i b^f + r_i$. Then
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

**Example:** Now take back $a_1 = 899900 \times 10^5$ and
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
representation. A floatting point number $(m, e) \in F$ with $e \geq -M$ where
$M$ is a positive integer is stored in memory (and registers) as the integer
$(\pm (e + M)b^p + |m|)$ with $|m| \geq b^{p - 1}$ if we consider
a non zero element of $F$.

**Proposition:** If $e_1 \geq e_2 + 2$ then $U(a_1, a_2) \geq b^p$.

**Proof:** We have $U(a_1, a_2) = |m_1 b^{e_1 - e_2} - m_2|
\geq ||m_1| b^{e_1 - e_2} - |m_2||$. But as $|m_1| \geq b^{p - 1}$ we have
$|m_1| b^{e_1 - e_2} \geq b^{p - 1 + e_1 - e_2} \geq p^{p + 1} > |m_2|$,
therefore $||m_1| b^{e_1 - e_2} - |m_2|| \geq |m_1|b^2 - |m_2|
> b^{p - 1 + 2} - b^p = b^p$.

The proprosition above basically states that if two floatting point numbers
are two orders of magnitude away then that have no digits in common, nothing
in common.

The usual definition of the distance in ULPs is usually given as the number
of floatting point numbers between the two considered floatting point numbers.
We will denote it by $V$ and can be easily calculated as follows:
- $V(a_1, a_2) = |(e_1 + M)b^p + m_1 - (e_2 + M)b^p - m_2|$ if $a_1$ and
  $a_2$ have the same signs
- $V(a_1, a_2) = V(a_1, 0) + V(a_2, 0)$ otherwise.

**Proposition:** If $e_1 = e_2$ and $a_1$, $a_2$ have the same sign then
$U(a_1, a_2) = V(a_1, a_2)$.

**Proof:** We have $V(a_1, a_2) = |(e_1 + M)b^p + m_1 - (e_2 + M)b^p - m_2|$,
but as $e_1 = e_2$, we end up with $V(a_1, a_2) = |m_1 - m_2| = U(a_1, a_2)$.

**Proposition:** $V(a_1, a_2) = 1$ is equivalent to $U(a_1, a_2) = 1$.

**Proof:** The proposition is true if $e_1 = e_2$. Suppose that $e_2 > e_1$,
and that $a_1, a_2 > 0$. We first suppose that $V(a_1, a_2) = 1$.
Then $e_2 = e_1 + 1$ otherwise we have that
$a_1 = m_1b^{e_1} < m_1b^{e_1 + 1} < m_2b^{e_1 + 2} \leq a_2$. Now we have
$(b^p - 1)b^{e_1} < b^{p - 1}b^{e_1 + 1}$ and let
$(b^p - 1)b^{e_1} \leq mb^e \leq b^{p - 1}b^{e_1 + 1}$. Then
$(b^p - 1)/m b^{e_1} < b^e < b^{p - 1}/m b^{e_1 + 1}$. But
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
$e_2 = e_1 + 1$. Then we have $b^p + m_2 - m_1 \geq b^p + b^{p - 1} - b^p$
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
$D(a_1, a_2) = U(a_1, a_2) / m_1$. Moreover we have
$b^{-p}U \leq D \leq b^{1 - p}U$.

**Proof:** Suppose that $|a_1| < |a_2|$, then $m_1/m_2 b^{e_1 - e_2} < 1$
but $m_1/m_2 \geq b^{p - 1} / b^p = b^{-1}$ and we have $b^{e_1 - e_2 - 1} < 1$
and $e_1 < e_2 + 1$ or $e_1 \leq e_2$. As a consequence we have
$D(a_1, a_2) = |m_1b^{e_1} - m_2b^{e_2}| / \min(m_1b^{e_1}, m_2b^{e_2})
= |m_1 - m_2b^{e_2 - e_1}| / \min(m_1, m_2b^{e_2 - e_1})$. Therefore
$D(a_1, a_2) = U(a_1, a_2) / \min(m_1, m_2b^{e_2 - e_1})$. Now if $e_1 = e_2$
then $\min(m_1, m_2) = m_1$ but if $e_2 > e_1$ then
$m_1 < b^p = b \times b^{p - 1} \leq b^{e_2 - e_1} m_2$ and again
$\min(m_1, m_2b^{e_2 - e_1}) = m_1$. Applying $b^{p - 1} \leq m_1 < b^p$ we get
that $b^{-p}U \leq D \leq b^{1 - p}U$.

**Remark:** Using the inequality of the previous proposition and taking the
base-$b$ logarithm we get $-p + \log U \leq \log D \leq 1 - p + \log U$ and
then $-p + \lfloor \log U \rfloor \leq \lfloor \log D \rfloor
\leq 1 - p + \lfloor \log U \rfloor$ hence two possibilities:
- $-p + \lfloor \log U \rfloor = \lfloor \log D \rfloor$ in which case
  $\lfloor \log U \rfloor + (-\lfloor \log D \rfloor) = p$.
- $1 - p + \lfloor \log U \rfloor = \lfloor \log D \rfloor$ in which case
  $1 + \lfloor \log U \rfloor + (-\lfloor \log D \rfloor) = p$.
According to a above proposition we know that $f = 1 + \lfloor \log U \rfloor$
can be interpreted as the number of differents digits in the last places of the
mantissa. Write $\mathcal{D} = - \lfloor \log D \rfloor$ then
$p \leq f + \mathcal{D} \leq p + 1$. The latter inequality shows that
$\mathcal{D}$ can be interpreted as the number of digits which are the same in
the mantissa near the "first" place. We can conclude this remark with the
interpretation that two floatting point numbers have at least $\mathcal{D} - 1$
digits in common in the first place of the mantissa and $f$ digits which are
different in the last place of the mantissa.
