# Four equivalent forms of the Fourier entropy-influence conjecture

Yitzchak Shmalo

This repository contains a short paper on the Fourier entropy-influence (FEI) conjecture of Friedgut and Kalai,
together with the script behind every numerical value in it.

Paper: [main.pdf](main.pdf) (source [main.tex](main.tex)).

## Summary

Let mu(S) = fhat(S)^2 be the spectral distribution of a Boolean function f and gamma(T) = sum over U containing T of
mu(U) the generalized influence of T. The Fourier entropy splits as H = A + B with A = E log2(1/gamma(S)) and
B = E log2(gamma(S)/mu(S)). The paper shows:

* (a) H <= 2B + 2I - 2Var, because derivatives of Boolean functions take values in a lattice. So FEI holds if and only
  if B <= C I.
* (b) For fixed t < 1, H lies between A_t and A_t + I log2(1/(1-t)), where A_t uses the noisy generalized influences
  ||T_sqrt(t) d_T f||^2. So FEI holds if and only if A_t <= C I.
* (c) B is an integral over the noise parameter of the expected number of elements a noise-tilted posterior adds to
  the spectral sample; only t in [1/2, 1] matters, and H <= 4I - 2Var + 2M(f) for every f.
* (d) FEI holds if and only if a uniformly chosen element of the size-biased spectral sample can be predicted from the
  others at a cost of O(1) bits, and this loss has a variational form over predictors.

The equivalences are for the class of all Boolean functions; for a function with odd Fourier support, such as
majority, the erasure entropies vanish while H/I does not.

## Code

The script needs Python 3 and NumPy. Its saved output is next to it.

* `code/check_forms.py` (output `check_forms_out.txt`): every identity and inequality of the theorem, the granularity
  lemma, the constant-predictor bound and the invariance under multiplication by a character, computed with exact
  Walsh-Hadamard transforms on 119 functions (random functions on 3 to 9 variables, AND_8, two tribes functions,
  majority on 9 variables).

## Related papers

* [The one-coordinate step in the Fourier entropy-influence conjecture](https://github.com/yspennstate/fourier-entropy-influence-one-coordinate)
* [Good coordinates in the Fourier entropy-influence conjecture](https://github.com/yspennstate/fei-good-coordinates)
* [A variance form of Han's bound on the Fourier entropy](https://github.com/yspennstate/fei-variance-han-bound)

## License

MIT, see [LICENSE](LICENSE).
