* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n4 0 n1 QNPNdefault
R1 n3 0 1k
R2 n1 n2 1k
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end
