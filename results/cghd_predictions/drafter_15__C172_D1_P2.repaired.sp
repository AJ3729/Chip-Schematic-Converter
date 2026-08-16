* Auto-generated SPICE netlist (NO TEXT OCR USED)

R1 n1 n2 1k
R2 n4 n2 1k
R3 n9 n7 1k
R4 n9 n8 1k
C1 n1 n7 1u
C2 n3 n8 1u
R5 n9 n10 1k
R6 n10 n2 1k
R7 n3 0 1k
C3 n11 n12 1u
R8 n8 n11 1k
C4 n10 n9 1u
I1 n6 n11 DC 1m
I2 n5 n10 DC 1m
C5 n4 n2 1u

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end
