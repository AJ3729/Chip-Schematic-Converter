* Auto-generated SPICE netlist (NO TEXT OCR USED)

C1 0 n5 1u
R1 0 n2 1k
R2 n4 0 1k
R3 0 n2 1k
R4 n4 0 1k
V1 n2 n5 DC 5
R5 n3 n2 1k
C2 0 n3 1u
V2 n1 n4 DC 5

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end
