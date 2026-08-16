* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 0 n3 n4 QNPNdefault
R1 n2 n3 1k
R2 n1 0 1k
.model QNPNdefault NPN

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09

.op
.end
