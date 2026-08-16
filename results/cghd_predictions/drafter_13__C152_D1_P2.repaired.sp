* Auto-generated SPICE netlist (NO TEXT OCR USED)

* UNSNAPPED Inductor raw_nodes=[0, None]
D1 n4 n2 Zdefault
C1 0 n3 1u
D2 n4 n2 Ddefault
C2 n1 n3 1u
D3 n3 n4 Ddefault
I1 n2 n3 AC 1m
.model Ddefault D
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end
