# Third-party notices

## Naibbe cipher

Voynich Structure Lab includes a compatibility implementation of the paper-facing Naibbe cipher and an exact copy of `references/naibbe_tables.csv` from:

- Repository: `greshko/naibbe-cipher`
- Pinned upstream commit: `f2675ec5dd275268bc64dd48ea64fc0e0e9827a2`
- Upstream baseline source: `naibbe.py`
- Imported table: `references/naibbe_tables.csv`
- Citation: Michael A. Greshko (2025), “The Naibbe cipher: a substitution cipher that encrypts Latin and Italian as Voynich Manuscript-like ciphertext,” *Cryptologia*. DOI: 10.1080/01611194.2025.2566408.

The implementation in this repository intentionally follows the paper-facing 52-card defaults from `naibbe.py`. The upstream `naibbe_v2.py` changes the default deck and ambiguity logic and is therefore treated as a separate variant rather than silently substituted for the baseline.

### Upstream license

MIT License

Copyright (c) 2025 greshko

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

The upstream README additionally requests citation of the Naibbe paper in publications that use substantial portions of the source or data; the citation is included above.
