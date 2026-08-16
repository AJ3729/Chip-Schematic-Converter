* Auto-generated SPICE netlist (NO TEXT OCR USED)

E1 n1 0 0 n3 100k
R1 n3 0 1k
* SAME_NODE_SKIPPED Capacitor both_on=0
R2 0 n3 1k
R3 0 n2 1k
R4 n4 0 1k
R5 0 n4 1k
* SAME_NODE_SKIPPED V-DC (one port) both_on=0
D1 n1 n2 Zdefault
* SAME_NODE_SKIPPED Zener Diode both_on=0
.model Zdefault D(bv=5.1)

.op
.end
