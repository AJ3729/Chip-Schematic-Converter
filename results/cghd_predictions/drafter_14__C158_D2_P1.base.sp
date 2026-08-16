* Auto-generated SPICE netlist (NO TEXT OCR USED)

D1 n4 n3 Zdefault
C1 0 n2 1u
* SAME_NODE_SKIPPED Zener Diode both_on=n2
C2 n2 0 1u
* SAME_NODE_SKIPPED Diode both_on=n2
C3 n2 n1 1u
Q1 n2 n2 n4 QNPNdefault
M1 n1 n2 n3 n3 PMOSdefault
.model PMOSdefault PMOS
.model QNPNdefault NPN
.model Zdefault D(bv=5.1)

.op
.end
