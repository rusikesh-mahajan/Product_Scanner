"""Pytest configuration and shared test fixtures."""

import os
import sys

import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def benetton_jeans_ocr_text():
    """Simulated OCR text for Tag 1: Benetton Jeans."""
    return """Size
WAIST : 80 cm
INSEAM LENGTH : 82 cm

MRP Rs. 3299.00
(incl. of all taxes)
Net Quantity : 1 N

Commodity: JEANS
Color : 902 Theme : A03
MENS / SUMMER

Marketed & Distributed by:
Benetton India Pvt. Ltd
B-25, Sector-34, Infocity,
Gurgaon -122001, Haryana (India)
Phone : +91-124-4590433

For Consumer Complaints Contact:
Asst. Manager Loyalty Programs
At Marketer's Address
Toll Free No.: 1800 102 4276
Email: c.services@benetton.co.in

8909329191647
26P4MBOO5043I - 30"""


@pytest.fixture
def jackjones_shirt_ocr_text():
    """Simulated OCR text for Tag 2: Jack & Jones Shirt."""
    return """JACK&JONES
Product Name: JJDI WAVORIA SHIRT LS
Generic Name: SHIRT
Style No.: 9017901005
Size: 1.06 m  M
Color: Cashmere Blue
Net Quantity-1N
MRP Rs. 3299.00 (incl. of all taxes)
Manufacturing Date:   Jul2025
Manufactured by: Carnation Creations Private Limited
3/61, Narasimhanaickenpalayam,
Coimbatore, 641031, 33, Tamilnadu
Marketed by: Best United India Comforts Private Limited, The Lalit
Residency, 2nd Floor, The Lalit
Mumbai, Sahar Airport Road, Sahar,
Andheri(E), Mumbai- 400059, 27,
Maharashtra

For customer complaints write
to address of marketing company
Email: help@jackjones.in
Customer Care Executive Ph. No.
1800260005
Country of Origin: India

8905944710232"""


@pytest.fixture
def benetton_polo_ocr_text():
    """Simulated OCR text for Tag 3: Benetton Polo."""
    return """Size
LENGTH : 71 cm
CHEST : 98 cm

MRP Rs. 2999.00
(incl. of all taxes)
Net Quantity : 1 N

Commodity: POLO T-SHIRT
Color : N22 Theme : C69
MENS / SUMMER

Marketed & Distributed by:
Benetton India Pvt. Ltd
B-25, Sector-34, Infocity,
Gurgaon -122001, Haryana (India)
Phone : +91-124-4590433

For Consumer Complaints Contact:
Asst. Manager Loyalty Programs
At Marketer's Address
Toll Free No.: 1800 102 4276
Email: c.services@benetton.co.in

8909329255677
26P3P59J3029I - M"""

