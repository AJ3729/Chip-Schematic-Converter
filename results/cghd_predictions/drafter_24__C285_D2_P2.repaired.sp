* Auto-generated SPICE netlist (NO TEXT OCR USED)

M1 0 n1 n2 n2 PMOSdefault
C1 0 n3 1u
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
M2 0 0 0 0 NMOSdefault
M3 0 0 0 0 PMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=0
.model NMOSdefault NMOS
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end
