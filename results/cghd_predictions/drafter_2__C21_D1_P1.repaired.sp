* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n7 n2 DC 5
C1 n3 n2 1u
C2 n6 n8 1u
R1 n4 n6 1k
C3 n5 0 1u
R2 n4 n5 1k
R3 n1 0 1k
R4 0 n5 1k
I1 0 n3 AC 1m
R5 n1 n3 1k
I2 0 n3 DC 1m
R6 n5 n9 1k

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09
Rshunt_n8 n8 0 1e+09
Rshunt_n9 n9 0 1e+09

.op
.end
