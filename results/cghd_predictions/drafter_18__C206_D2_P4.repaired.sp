* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n1 0 Zdefault
D2 0 n3 Zdefault
D3 0 n2 Zdefault
* UNSNAPPED Capacitor raw_nodes=[0, None]
D4 n4 0 Zdefault
* SAME_NODE_SKIPPED Diode both_on=0
M1 n4 0 0 0 PMOSdefault
.model PMOSdefault PMOS
.model Zdefault D(bv=5.1)

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09
Rshunt_n2 n2 0 1e+09
Rshunt_n3 n3 0 1e+09

.op
.end
