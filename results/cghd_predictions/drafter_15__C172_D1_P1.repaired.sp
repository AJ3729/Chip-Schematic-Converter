* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n4 n7 1k
C1 n9 n10 1u
R2 n1 n8 1k
C2 n8 n11 1u
R3 n10 n8 1k
R4 n1 n2 1k
R5 n11 0 1k
R6 n1 n3 1k
C3 n3 n4 1u
R7 n2 n7 1k
I1 n10 n12 AC 1m
R8 n6 n7 1k
I2 n5 n2 DC 1m
V1 n1 0 DC 5
C4 n1 n2 1u
C5 n6 n7 1u
I3 n2 n5 AC 1m
I4 n12 n10 DC 1m

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end
