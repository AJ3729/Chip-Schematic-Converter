* Auto-generated SPICE netlist (NO TEXT OCR USED)

Q1 n2 n10 n10 QNPNdefault
Q2 n25 n7 n26 QPNPdefault
Q3 n25 n27 n7 QPNPdefault
* SAME_NODE_SKIPPED Diode both_on=n7
Q4 n2 n9 n10 QNPNdefault
Q5 n7 n29 n31 QNPNdefault
D1 n7 n2 Ddefault
D2 n12 n7 Ddefault
* SAME_NODE_SKIPPED Diode both_on=n7
R1 n1 n2 1k
D3 n25 n19 Ddefault
R2 n10 n7 1k
Q6 n4 n3 n1 QPNPdefault
* SAME_NODE_SKIPPED Resistor both_on=n7
Q7 n2 n8 n11 QNPNdefault
R3 n37 n25 1k
Q8 n25 n26 n7 QPNPdefault
R4 n7 n28 1k
Q9 n7 n7 n7 QPNPdefault
R5 n10 n7 1k
Q10 n7 n18 n7 QNPNdefault
* SAME_NODE_SKIPPED Resistor both_on=n7
* SAME_NODE_SKIPPED Resistor both_on=n10
* SAME_NODE_SKIPPED Resistor both_on=n7
R6 n1 n4 1k
R7 n7 n30 1k
C1 n25 n38 1u
R8 n31 n33 1k
R9 n13 n4 1k
* SAME_NODE_SKIPPED Resistor both_on=n7
R10 n9 n7 1k
R11 n7 n21 1k
R12 n33 n36 1k
R13 n7 n26 1k
R14 n4 n3 1k
* SAME_NODE_SKIPPED Resistor both_on=n7
R15 n24 n35 1k
* SAME_NODE_SKIPPED Resistor both_on=n7
R16 n11 n7 1k
C2 n23 0 1u
R17 n5 n7 1k
R18 n7 n27 1k
R19 n13 n7 1k
* SAME_NODE_SKIPPED Diode both_on=n7
Q11 n12 n7 n7 QNPNdefault
R20 n1 n4 1k
Q12 n3 n7 n4 QPNPdefault
Q13 n4 n24 n22 QPNPdefault
R21 n4 n24 1k
R22 n1 n3 1k
C3 n4 n13 1u
Q14 n36 n33 n29 QNPNdefault
R23 n17 n15 1k
R24 n15 n22 1k
Q15 n22 n24 n22 QPNPdefault
Q16 n2 n7 n9 QNPNdefault
R25 n24 n34 1k
R26 n8 n10 1k
Q17 n4 n4 n4 QPNPdefault
R27 n10 n11 1k
R28 n14 n7 1k
C4 n25 n38 1u
* SAME_NODE_SKIPPED Resistor both_on=n4
C5 n25 n16 1u
R29 n6 n4 1k
R30 n20 n7 1k
C6 n4 n7 1u
C7 n23 0 1u
R31 n14 n18 1k
* SAME_NODE_SKIPPED Capacitor both_on=n7
R32 n18 n20 1k
* SAME_NODE_SKIPPED Resistor both_on=n4
* SAME_NODE_SKIPPED Resistor both_on=n7
Q18 n25 n28 n7 QPNPdefault
C8 n24 n32 1u
C9 n10 n11 1u
* SAME_NODE_SKIPPED Resistor both_on=n17
R33 n4 n23 1k
R34 n4 n15 1k
* SAME_NODE_SKIPPED Capacitor both_on=n2
C10 n1 n2 1u
R35 n4 n15 1k
C11 n24 n29 1u
C12 n4 n7 1u
* SAME_NODE_SKIPPED Capacitor both_on=n7
R36 n4 n24 1k
* SAME_NODE_SKIPPED Diode both_on=n7
* SAME_NODE_SKIPPED Resistor both_on=n4
* SAME_NODE_SKIPPED Capacitor both_on=n1
* SAME_NODE_SKIPPED Diode both_on=n7
.model Ddefault D
.model QNPNdefault NPN
.model QPNPdefault PNP

* --- design-intent repair (does not change topology) ---
Rshunt_n1 n1 0 1e+09

.op
.end
