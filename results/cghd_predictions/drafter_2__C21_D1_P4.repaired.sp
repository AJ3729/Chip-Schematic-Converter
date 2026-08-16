* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n4 n2 DC 5
R1 n5 n7 1k
R2 0 n6 1k
C1 n7 n4 1u
C2 0 n6 1u
C3 n1 n3 1u
R3 n1 0 1k
I1 0 n1 AC 1m
R4 n5 n6 1k
V2 0 n1 AC 1

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end
