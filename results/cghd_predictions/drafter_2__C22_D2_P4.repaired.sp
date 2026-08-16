* Auto-generated SPICE netlist (NO TEXT OCR USED)

V1 n2 0 DC 5
D1 0 n1 Ddefault
M1 n1 0 n1 n1 NMOSdefault
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
M2 0 0 0 0 PMOSdefault
* SAME_NODE_SKIPPED Capacitor both_on=0
M3 n1 0 0 0 PMOSdefault
* SAME_NODE_SKIPPED I-DC both_on=0
M4 0 0 0 0 NMOSdefault
E1 0 0 0 0 100k
.model Ddefault D
.model NMOSdefault NMOS
.model PMOSdefault PMOS

* --- design-intent repair (does not change topology) ---
Rshunt_n2 n2 0 1e+09

.op
.end
