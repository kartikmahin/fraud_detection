"""
Receipt Feature Extraction & General Feature Imputation Module
Extracts transaction features from uploaded receipts/bills (images, PDFs, text, JSON, CSV)
and imputes general baseline features for any missing inputs required by the MLP fraud model.
"""

import csv
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple


# General baseline features derived from the training dataset statistics (models/scaler.pkl)
# Used whenever a feature is not present on an uploaded bill/receipt.
GENERAL_FEATURE_DEFAULTS: Dict[str, Any] = {
    "transaction_amount": 150.0,      # Typical legitimate transaction amount (INR)
    "transaction_time": 50400,        # 14:00 (2:00 PM) - training set median hour (14)
    "location": "Mumbai",             # Mode city in the training distribution
    "device_id": "DEV001",            # General default device identifier
    "merchant_category": "retail",    # Mode merchant category in the training distribution
    "account_age_days": 1837,         # Training set median account age in days (~5 years)
    "transaction_count_24h": 5,       # Training set median 24-hour transaction count
    "avg_transaction_amount": 150.0,  # Typical legitimate customer average transaction amount (INR)
}

VALID_LOCATIONS = [
    "Mumbai", "Delhi", "Bengaluru", "Hyderabad",
    "Chennai", "Kolkata", "Pune", "Ahmedabad",
]

VALID_MERCHANT_CATEGORIES = [
    "retail", "grocery", "restaurant", "gas", "online",
]

# City aliases and regional keywords mapped to the 8 canonical model locations
LOCATION_ALIASES: Dict[str, List[str]] = {
    "Mumbai": [
        "mumbai", "bombay", "navi mumbai", "thane", "andheri", "bandra",
        "powai", "worli", "colaba", "borivali", "dadar", "kurla", "malad",
        "goregaon", "vashi", "panvel", "juhu", "lower parel", "mulund", "ghatkopar",
    ],
    "Delhi": [
        "delhi", "new delhi", "ncr", "noida", "gurgaon", "gurugram",
        "faridabad", "ghaziabad", "connaught place", "saket", "dwarka",
        "rohini", "lajpat nagar", "karol bagh", "vasant kunj", "hauz khas",
    ],
    "Bengaluru": [
        "bengaluru", "bangalore", "whitefield", "koramangala", "indiranagar",
        "electronic city", "jayanagar", "marathahalli", "hsr layout",
        "hebbal", "bellandur", "malleshwaram", "mg road bangalore", "yelahanka",
    ],
    "Hyderabad": [
        "hyderabad", "secunderabad", "hitec city", "hitech city", "gachibowli",
        "banjara hills", "jubilee hills", "madhapur", "kukatpally", "ameerpet",
        "begumpet", "kondapur", "shamshabad",
    ],
    "Chennai": [
        "chennai", "madras", "anna nagar", "t nagar", "t. nagar", "velachery",
        "adyar", "guindy", "mylapore", "tambram", "tambaram", "nungambakkam",
        "egmore", "porur",
    ],
    "Kolkata": [
        "kolkata", "calcutta", "salt lake", "bidhannagar", "new town",
        "park street", "howrah", "dum dum", "ballygunge", "alipore", "gariahat",
    ],
    "Pune": [
        "pune", "pimpri", "chinchwad", "hinjewadi", "kothrud", "viman nagar",
        "wakad", "baner", "hadapsar", "magarpatta", "koregaon park", "aundh",
        "shivajinagar", "kharadi",
    ],
    "Ahmedabad": [
        "ahmedabad", "amdavad", "gandhinagar", "navrangpura", "satellite",
        "vastrapur", "prahlad nagar", "maninagar", "SG highway", "bopal", "cg road",
    ],
}

# 3-digit Indian PIN code prefixes mapped to canonical cities
PINCODE_PREFIX_TO_CITY: Dict[str, str] = {
    "400": "Mumbai",
    "401": "Mumbai",
    "410": "Mumbai",
    "110": "Delhi",
    "121": "Delhi",
    "122": "Delhi",
    "201": "Delhi",
    "560": "Bengaluru",
    "500": "Hyderabad",
    "600": "Chennai",
    "700": "Kolkata",
    "711": "Kolkata",
    "411": "Pune",
    "412": "Pune",
    "380": "Ahmedabad",
    "382": "Ahmedabad",
}

# GSTIN 2-digit state codes mapped to default canonical city in that state
GSTIN_STATE_TO_CITY: Dict[str, str] = {
    "27": "Mumbai",     # Maharashtra
    "07": "Delhi",      # Delhi
    "06": "Delhi",      # Haryana (Gurugram/Faridabad)
    "09": "Delhi",      # UP (Noida/Ghaziabad)
    "29": "Bengaluru",  # Karnataka
    "36": "Hyderabad",  # Telangana
    "33": "Chennai",    # Tamil Nadu
    "19": "Kolkata",    # West Bengal
    "24": "Ahmedabad",  # Gujarat
}

# Keywords for inferring merchant_category from bill/receipt text
MERCHANT_CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "grocery": [
        "grocery", "groceries", "supermarket", "hypermarket", "dmart", "d-mart",
        "avenue supermarts", "bigbasket", "reliance fresh", "reliance smart",
        "more supermarket", "nature's basket", "blinkit", "zepto", "instamart",
        "spencer's", "star bazaar", "kirana", "provisions", "departmental store",
        "vegetables", "fruits", "dairy", "milk", "curd", "paneer", "bread",
        "eggs", "atta", "wheat", "rice", "dal", "pulses", "cooking oil",
        "spices", "masala", "sugar", "salt", "biscuits", "fresh produce",
    ],
    "restaurant": [
        "restaurant", "cafe", "coffee", "bistro", "diner", "dining", "kitchen",
        "eatery", "food court", "zomato", "swiggy", "starbucks", "mcdonald",
        "domino", "pizza", "burger", "kfc", "subway", "haldiram", "barbeque",
        "bbq", "bakery", "pastry", "sweet", "dhaba", "bar & grill", "lounge",
        "waiter", "table no", "dine-in", "dine in", "takeaway", "take away",
        "fssai", "food bill", "beverage", "appetizer", "entree", "dessert",
        "biryani", "tandoori", "naan", "roti", "curry", "latte", "cappuccino",
        "espresso", "snacks", "meal", "buffet", "cover charge", "service charge",
    ],
    "gas": [
        "petrol", "diesel", "fuel", "gas station", "cng", "lpg", "indian oil",
        "iocl", "bharat petroleum", "bpcl", "hindustan petroleum", "hpcl",
        "shell", "nayara", "essar oil", "reliance petroleum", "filling station",
        "service station", "fuel pump", "petrol pump", "nozzle", "litre",
        "litres", "ltr", "ltrs", "density", "rate/ltr", "vehicle no",
        "unleaded", "power fuel", "speed petrol",
    ],
    "online": [
        "online", "e-commerce", "ecommerce", "amazon", "flipkart", "myntra",
        "ajio", "meesho", "nykaa", "tatacliq", "tata cliq", "snapdeal",
        "jiomart", "shopify", "ebay", "paypal", "stripe", "razorpay", "payu",
        "cashfree", "ccavenue", "billdesk", "order id", "order #", "order no",
        "awb", "tracking id", "shipping", "delivery fee", "delivery charge",
        "shipping address", "billing address", "payment gateway", "digital",
        "subscription", "cloud", "app store", "play store", "steam", "netflix",
        "spotify", "uber", "ola", "makemytrip", "irctc", "bookmyshow",
    ],
    "retail": [
        "retail", "store", "shop", "mall", "emporium", "boutique", "outlet",
        "apparel", "fashion", "clothing", "garments", "footwear", "shoes",
        "electronics", "croma", "reliance digital", "vijay sales", "lifestyle",
        "shoppers stop", "pantaloons", "westside", "zudio", "trent", "max fashion",
        "titan", "tanishq", "kalyan", "malabar", "decathlon", "ikea", "crossword",
        "archies", "miniso", "pharmacy", "medical", "apollo", "medplus",
        "wellness", "cosmetics", "hardware", "furniture", "accessories",
        "jewellery", "watch", "optical", "lenskart", "Raymond", "fabindia",
    ],
}


def extract_text_from_file(
    file_bytes: bytes,
    filename: str = "",
    content_type: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """
    Extract raw text and any directly structured fields from an uploaded receipt file.
    Supports images (PNG/JPG/WEBP/BMP/TIFF), PDFs, JSON, CSV, HTML, and plain text.
    """
    if not file_bytes:
        return "", {}

    fname_lower = (filename or "").lower().strip()
    ctype_lower = (content_type or "").lower().strip()

    # 1. Check if JSON
    if fname_lower.endswith(".json") or "application/json" in ctype_lower:
        try:
            text_content = file_bytes.decode("utf-8", errors="ignore")
            parsed_json = json.loads(text_content)
            if isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict):
                parsed_json = parsed_json[0]
            if isinstance(parsed_json, dict):
                return text_content, _normalize_structured_dict(parsed_json)
        except Exception:
            pass

    # 2. Check if CSV
    if fname_lower.endswith(".csv") or "text/csv" in ctype_lower:
        try:
            text_content = file_bytes.decode("utf-8", errors="ignore")
            reader = list(csv.DictReader(io.StringIO(text_content)))
            if reader:
                return text_content, _normalize_structured_dict(reader[0])
        except Exception:
            pass

    # 3. Check if PDF (by extension, content-type, or magic header %PDF)
    if (
        fname_lower.endswith(".pdf")
        or "application/pdf" in ctype_lower
        or file_bytes[:4] == b"%PDF"
    ):
        pdf_text = _extract_text_from_pdf(file_bytes)
        return pdf_text, {}

    # 4. Check if Image (by extension, content-type, or magic bytes)
    is_image_magic = (
        file_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        or file_bytes[:3] == b"\xff\xd8\xff"
        or file_bytes[:4] in (b"RIFF", b"GIF8", b"II*\x00", b"MM\x00*")
        or file_bytes[:2] == b"BM"
    )
    if (
        any(fname_lower.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif"))
        or ctype_lower.startswith("image/")
        or is_image_magic
    ):
        img_text, meta_fields = _extract_text_from_image(file_bytes)
        return img_text, meta_fields

    # 5. Fallback: treat as UTF-8 text / HTML / Markdown
    try:
        text_content = file_bytes.decode("utf-8", errors="ignore")
        stripped = text_content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed_json = json.loads(stripped)
                if isinstance(parsed_json, dict):
                    return text_content, _normalize_structured_dict(parsed_json)
            except Exception:
                pass
        if "<html" in text_content.lower() or "<body" in text_content.lower() or "<table" in text_content.lower():
            text_content = re.sub(r"<[^>]+>", " ", text_content)
        return text_content, {}
    except Exception:
        return "", {}


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using pypdf, pdfplumber, or OCR fallback."""
    texts: List[str] = []

    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
    except Exception:
        pass

    if not texts:
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t.strip():
                        texts.append(t)
        except Exception:
            pass

    combined = "\n".join(texts).strip()
    if len(combined) >= 15:
        return combined

    # Scanned PDF fallback: render page via pypdfium2 and run OCR
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(file_bytes)
        ocr_texts = []
        for i in range(min(len(pdf), 3)):
            page = pdf[i]
            pil_image = page.render(scale=2.0).to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            t, _ = _extract_text_from_image(buf.getvalue())
            if t.strip():
                ocr_texts.append(t)
        if ocr_texts:
            return "\n".join(ocr_texts).strip()
    except Exception:
        pass

    return combined


def _extract_text_from_image(file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    """Extract text from receipt image using PIL + pytesseract with upscaling and contrast preprocessing."""
    meta_fields: Dict[str, Any] = {}
    ocr_chunks: List[str] = []

    try:
        from PIL import Image, ImageOps, ImageEnhance
        img = Image.open(io.BytesIO(file_bytes))

        # Check embedded image metadata (e.g. PNG text chunks or EXIF comments)
        if hasattr(img, "info") and isinstance(img.info, dict):
            for k, v in img.info.items():
                if isinstance(v, str) and len(v) > 2:
                    ocr_chunks.append(f"{k}: {v}")
                    if v.strip().startswith("{") and v.strip().endswith("}"):
                        try:
                            meta_fields.update(_normalize_structured_dict(json.loads(v)))
                        except Exception:
                            pass

        # Run pytesseract OCR
        try:
            import pytesseract
            gray = ImageOps.grayscale(img)
            # Upscale low-resolution receipt images for accurate OCR
            if gray.width < 1200:
                scale = 2
                resample = getattr(Image, "Resampling", Image).LANCZOS
                gray_up = gray.resize((gray.width * scale, gray.height * scale), resample)
            else:
                gray_up = gray

            text1 = pytesseract.image_to_string(gray_up) or ""
            if text1.strip():
                ocr_chunks.append(text1)

            text2 = pytesseract.image_to_string(gray) or ""
            if text2.strip() and text2.strip() != text1.strip():
                ocr_chunks.append(text2)

            if len(text1.strip()) < 40:
                enhanced = ImageEnhance.Contrast(gray_up).enhance(2.2)
                text3 = pytesseract.image_to_string(enhanced) or ""
                if text3.strip():
                    ocr_chunks.append(text3)
        except Exception:
            pass
    except Exception:
        pass

    return "\n".join(ocr_chunks).strip(), meta_fields


def _normalize_structured_dict(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a JSON/CSV dictionary into valid model input features."""
    extracted: Dict[str, Any] = {}
    lower_map = {str(k).strip().lower(): v for k, v in raw_dict.items() if v is not None and str(v).strip() != ""}

    # 1. transaction_amount
    for key in ("transaction_amount", "amount", "total", "grand_total", "bill_amount", "total_amount", "net_amount", "amount_paid"):
        if key in lower_map:
            val = _parse_float(lower_map[key])
            if val is not None and val > 0:
                extracted["transaction_amount"] = round(val, 2)
                break

    # 2. transaction_time
    if "transaction_time" in lower_map:
        val = lower_map["transaction_time"]
        parsed_sec = _parse_time_value(val)
        if parsed_sec is not None:
            extracted["transaction_time"] = parsed_sec
    elif "time" in lower_map or "timestamp" in lower_map or "date" in lower_map:
        for key in ("time", "timestamp", "date"):
            if key in lower_map:
                parsed_sec = _parse_time_value(lower_map[key])
                if parsed_sec is not None:
                    extracted["transaction_time"] = parsed_sec
                    break
    elif "hour_of_day" in lower_map or "hour" in lower_map:
        hr = _parse_float(lower_map.get("hour_of_day", lower_map.get("hour")))
        if hr is not None and 0 <= int(hr) <= 23:
            extracted["transaction_time"] = int(hr) * 3600

    # 3. location
    for key in ("location", "city", "branch", "store_location", "place", "address"):
        if key in lower_map:
            loc = _extract_location_from_text(str(lower_map[key]))
            if loc:
                extracted["location"] = loc
                break

    # 4. merchant_category
    for key in ("merchant_category", "category", "merchant_type", "type", "merchant", "store_name", "vendor"):
        if key in lower_map:
            cat = _extract_merchant_category_from_text(str(lower_map[key]))
            if cat:
                extracted["merchant_category"] = cat
                break

    # 5. device_id
    for key in ("device_id", "device", "terminal_id", "tid", "pos_id"):
        if key in lower_map:
            dev_str = str(lower_map[key]).strip()
            if dev_str:
                extracted["device_id"] = dev_str
                break

    # 6. account_age_days
    for key in ("account_age_days", "account_age", "customer_tenure_days", "tenure_days"):
        if key in lower_map:
            val = _parse_float(lower_map[key])
            if val is not None and val >= 0:
                extracted["account_age_days"] = int(round(val))
                break

    # 7. transaction_count_24h
    for key in ("transaction_count_24h", "txn_count_24h", "transactions_24h", "daily_transactions", "transaction_count"):
        if key in lower_map:
            val = _parse_float(lower_map[key])
            if val is not None and val >= 0:
                extracted["transaction_count_24h"] = int(round(val))
                break

    # 8. avg_transaction_amount
    for key in ("avg_transaction_amount", "average_transaction_amount", "avg_amount", "avg_spend", "average_amount"):
        if key in lower_map:
            val = _parse_float(lower_map[key])
            if val is not None and val > 0:
                extracted["avg_transaction_amount"] = round(val, 2)
                break

    return extracted


def _parse_float(val: Any) -> Optional[float]:
    """Parse a float from a string or number, stripping currency symbols and commas."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return None
    cleaned = re.sub(r"[₹$,\s]|(?i:inr|rs\.?|usd)", "", val.strip())
    m = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def _parse_time_value(val: Any) -> Optional[int]:
    """Parse either integer seconds-since-midnight or a time/timestamp string."""
    if isinstance(val, (int, float)):
        ival = int(round(val))
        if 0 <= ival <= 86399:
            return ival
    if isinstance(val, str):
        s = val.strip()
        if re.fullmatch(r"\d{1,5}", s):
            ival = int(s)
            if 0 <= ival <= 86399:
                return ival
        return _extract_time_from_text(s)
    return None


def _extract_amount_from_text(text: str) -> Optional[float]:
    """
    Extract the total transaction amount from receipt text.
    Prioritizes explicit total labels (including OCR-tolerant variants like 'Grand Tota'),
    then falls back to Subtotal or currency/decimal prices.
    """
    if not text or not text.strip():
        return None

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Tier 1: Explicit final total labels on a line (OCR-tolerant for Total / Amount / Paid)
    tier1_pattern = re.compile(
        r"(?i)\b(?:grand\s*tota[l1i]?|tota[l1i]?\s*am[o0]unt|am[o0]unt\s*pa[i1l]d|net\s*am[o0]unt|b[i1l]{2}\s*am[o0]unt|"
        r"tota[l1i]?\s*payable|net\s*payable|order\s*tota[l1i]?|transaction\s*am[o0]unt|tota[l1i]?\s*due|"
        r"balance\s*due|am[o0]unt\s*payable|final\s*am[o0]unt|pa[i1l]d\s*am[o0]unt|tota[l1i]?\s*pa[i1l]d)\b"
        r"[^\d\n]{0,25}?(?:₹|rs\.?|ps\.?|inr|\$)?\s*([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)"
    )
    tier1_matches: List[float] = []
    for line in lines:
        for m in tier1_pattern.finditer(line):
            val = _parse_float(m.group(1))
            if val is not None and 0 < val < 10_000_000:
                tier1_matches.append(val)
    if tier1_matches:
        return round(tier1_matches[-1], 2)

    # Tier 2: Generic "Total" or "Amount" label (excluding "Sub Total", "Tax Total", "Total Items", "Avg")
    tier2_pattern = re.compile(
        r"(?i)(?<!sub\s)(?<!sub-)(?<!tax\s)(?<!gst\s)(?<!avg\s)(?<!average\s)"
        r"\b(?:tota[l1i]?|am[o0]unt)\b(?!\s*(?:items|qty|quantity|units|tax|gst|discount|saved|savings|points))"
        r"[^\d\n]{0,25}?(?:₹|rs\.?|ps\.?|inr|\$)?\s*([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)"
    )
    tier2_matches: List[float] = []
    for line in lines:
        if re.search(r"(?i)\b(avg|average|sub\s*total|subtotal|tax|cgst|sgst|igst|discount|qty|items)\b", line):
            continue
        for m in tier2_pattern.finditer(line):
            val = _parse_float(m.group(1))
            if val is not None and 1.0 <= val < 10_000_000:
                tier2_matches.append(val)
    if tier2_matches:
        return round(tier2_matches[-1], 2)

    # Tier 3: Multi-line Total (where "Grand Total" or "Total" is on one line and the amount is on the very next line)
    for i in range(len(lines) - 1):
        if re.fullmatch(r"(?i)(?:grand\s*tota[l1i]?|tota[l1i]?\s*amount|amount\s*paid|net\s*amount|tota[l1i]?\s*payable|tota[l1i]?)\s*:?", lines[i]):
            m = re.search(r"(?:₹|rs\.?|ps\.?|inr|\$)?\s*([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)", lines[i + 1], re.I)
            if m:
                val = _parse_float(m.group(1))
                if val is not None and 1.0 <= val < 10_000_000:
                    return round(val, 2)

    # Tier 4: Subtotal if no final total found
    subtotal_pattern = re.compile(
        r"(?i)\b(?:sub\s*tota[l1i]?|subtota[l1i]?)\b[^\d\n]{0,25}?(?:₹|rs\.?|ps\.?|inr|\$)?\s*([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)"
    )
    for line in lines:
        m = subtotal_pattern.search(line)
        if m:
            val = _parse_float(m.group(1))
            if val is not None and 1.0 <= val < 10_000_000:
                return round(val, 2)

    # Tier 5: Currency-prefixed numbers anywhere in the receipt (pick the max as total)
    currency_pattern = re.compile(
        r"(?i)(?:₹|rs\.?\s*|inr\s*|\$\s*)([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)"
    )
    currency_vals: List[float] = []
    for m in currency_pattern.finditer(text):
        val = _parse_float(m.group(1))
        if val is not None and 1.0 <= val < 10_000_000:
            currency_vals.append(val)
    if currency_vals:
        return round(max(currency_vals), 2)

    # Tier 6: Explicit decimal amounts (e.g. 280.00) on item/summary lines
    decimal_vals: List[float] = []
    for m in re.finditer(r"\b([0-9]{1,3}(?:,[0-9]{2,3})+|\d+)\.\d{2}\b", text):
        val = _parse_float(m.group(0))
        if val is not None and 1.0 <= val < 10_000_000:
            decimal_vals.append(val)
    if decimal_vals:
        return round(max(decimal_vals), 2)

    return None


def _extract_time_from_text(text: str) -> Optional[int]:
    """
    Extract transaction time from receipt text and convert to seconds since midnight (0-86399).
    Supports:
    - Explicit `transaction_time: 43200` or `hour_of_day: 14`
    - 12-hour time with AM/PM (e.g. `02:35 PM`, `11:20:15 AM`)
    - 24-hour time with label or date context (e.g. `Time: 18:45`, `19/10/2026 14:30`)
    """
    if not text or not text.strip():
        return None

    # 1. Explicit transaction_time in seconds
    m_sec = re.search(r"(?i)\btransaction[_\s]*time\s*[:=]\s*(\d{1,5})\b", text)
    if m_sec:
        sec = int(m_sec.group(1))
        if 0 <= sec <= 86399:
            return sec

    # 2. Explicit hour_of_day
    m_hr = re.search(r"(?i)\bhour[_\s]*of[_\s]*day\s*[:=]\s*(\d{1,2})\b", text)
    if m_hr:
        hr = int(m_hr.group(1))
        if 0 <= hr <= 23:
            return hr * 3600

    # 3. 12-hour format with AM/PM (e.g., "02:30 PM", "11:15:40 am", "9.45 PM")
    m_12h = re.search(
        r"\b(0?[1-9]|1[0-2])[:.]([0-5]\d)(?:[:.]([0-5]\d))?\s*([AaPp]\.?[Mm]\.?)\b",
        text,
    )
    if m_12h:
        hh = int(m_12h.group(1))
        mm = int(m_12h.group(2))
        ss = int(m_12h.group(3)) if m_12h.group(3) else 0
        meridiem = m_12h.group(4).lower().replace(".", "")
        if meridiem == "pm" and hh != 12:
            hh += 12
        elif meridiem == "am" and hh == 12:
            hh = 0
        return hh * 3600 + mm * 60 + ss

    # 4. Labeled 24-hour format (e.g., "Time: 14:35", "Time - 23:10:05", "At 08:15 hrs")
    m_labeled_24h = re.search(
        r"(?i)\b(?:time|timestamp|at|hrs)\s*[:=\-]?\s*([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b",
        text,
    )
    if m_labeled_24h:
        hh = int(m_labeled_24h.group(1))
        mm = int(m_labeled_24h.group(2))
        ss = int(m_labeled_24h.group(3)) if m_labeled_24h.group(3) else 0
        return hh * 3600 + mm * 60 + ss

    # 5. Date followed by 24-hour time (e.g., "19/10/2026 14:30", "2026-10-19 22:15:00", "19-Oct-2026 09:45")
    m_date_time = re.search(
        r"\b(?:\d{1,4}[/\-\.](?:\d{1,2}|[A-Za-z]{3,9})[/\-\.]\d{2,4})[\s,T]+([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b",
        text,
    )
    if m_date_time:
        hh = int(m_date_time.group(1))
        mm = int(m_date_time.group(2))
        ss = int(m_date_time.group(3)) if m_date_time.group(3) else 0
        return hh * 3600 + mm * 60 + ss

    # 6. Standalone HH:MM:SS or HH:MM with IST/HRS suffix
    m_24h_general = re.search(
        r"\b([01]\d|2[0-3]):([0-5]\d)(?::([0-5]\d)|\s*(?i:hrs|ist))\b",
        text,
    )
    if m_24h_general:
        hh = int(m_24h_general.group(1))
        mm = int(m_24h_general.group(2))
        ss = int(m_24h_general.group(3)) if m_24h_general.group(3) else 0
        return hh * 3600 + mm * 60 + ss

    return None


def _extract_location_from_text(text: str) -> Optional[str]:
    """
    Extract and normalize the city location from receipt text to one of the 8 model cities:
    Mumbai, Delhi, Bengaluru, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad.
    """
    if not text or not text.strip():
        return None

    text_lower = text.lower()

    # 1. Check explicit Location / City / Branch / Address lines first
    labeled_loc = re.search(
        r"(?i)\b(?:location|city|branch|store\s*location|place|address)\s*[:=\-]\s*([^\n\r]+)",
        text,
    )
    if labeled_loc:
        line_lower = labeled_loc.group(1).lower()
        for city, aliases in LOCATION_ALIASES.items():
            for alias in aliases:
                if re.search(rf"\b{re.escape(alias.lower())}\b", line_lower):
                    return city

    # 2. Scan full text for exact city names or locality aliases
    for city, aliases in LOCATION_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias.lower())}\b", text_lower):
                return city

    # 3. Check 6-digit Indian PIN codes (e.g., 400001 -> Mumbai, 560001 -> Bengaluru)
    for pin_match in re.finditer(r"\b([1-8]\d{2})\s?\d{3}\b", text):
        prefix = pin_match.group(1)
        if prefix in PINCODE_PREFIX_TO_CITY:
            return PINCODE_PREFIX_TO_CITY[prefix]

    # 4. Check 15-character Indian GSTIN state prefix (e.g. 27AAAAA0000A1Z5 -> Mumbai)
    gstin_match = re.search(r"\b(\d{2})[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b", text.upper())
    if gstin_match:
        state_code = gstin_match.group(1)
        if state_code in GSTIN_STATE_TO_CITY:
            return GSTIN_STATE_TO_CITY[state_code]

    return None


def _extract_merchant_category_from_text(text: str) -> Optional[str]:
    """
    Extract or infer the merchant category from receipt text, normalized to:
    retail, grocery, restaurant, gas, online.
    """
    if not text or not text.strip():
        return None

    text_lower = text.lower()

    # 1. Direct exact category name if text is just the category or has an explicit label
    stripped = text_lower.strip()
    if stripped in VALID_MERCHANT_CATEGORIES:
        return stripped

    labeled_cat = re.search(
        r"(?i)\b(?:merchant[_\s]*category|category|merchant[_\s]*type|store[_\s]*type)\s*[:=\-]\s*([A-Za-z]+)",
        text,
    )
    if labeled_cat:
        cat_candidate = labeled_cat.group(1).strip().lower()
        if cat_candidate in VALID_MERCHANT_CATEGORIES:
            return cat_candidate
        for cat, keywords in MERCHANT_CATEGORY_KEYWORDS.items():
            if cat_candidate in keywords:
                return cat

    # 2. Score categories by keyword matches in the receipt text
    scores: Dict[str, int] = {cat: 0 for cat in VALID_MERCHANT_CATEGORIES}
    for cat, keywords in MERCHANT_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw.lower())}\b", text_lower):
                weight = 3 if kw.lower() in (cat, "supermarket", "hypermarket", "restaurant", "cafe", "petrol", "diesel", "fuel", "amazon", "flipkart", "zomato", "swiggy", "dmart", "bigbasket", "indian oil", "bpcl", "hpcl") else 1
                scores[cat] += weight

    best_cat = max(scores, key=lambda k: scores[k])
    if scores[best_cat] > 0:
        return best_cat

    return None


def _extract_optional_banking_fields(text: str) -> Dict[str, Any]:
    """
    Extract optional banking/account metadata if printed on the receipt or transaction slip:
    - device_id
    - account_age_days
    - transaction_count_24h
    - avg_transaction_amount
    """
    extra: Dict[str, Any] = {}
    if not text or not text.strip():
        return extra

    # device_id (e.g. "Device ID: DEV012", "Terminal ID: POS402", or "DEV005")
    m_dev = re.search(
        r"(?i)\b(?:device[_\s]*id|terminal[_\s]*id|pos[_\s]*id|tid)\s*[:=\-]\s*([A-Za-z0-9_\-]+)",
        text,
    )
    if m_dev:
        extra["device_id"] = m_dev.group(1).strip()
    else:
        m_dev_code = re.search(r"\b(DEV\d{2,5})\b", text)
        if m_dev_code:
            extra["device_id"] = m_dev_code.group(1)

    # account_age_days (e.g. "Account Age: 365 days", "account_age_days: 45")
    m_age = re.search(
        r"(?i)\b(?:account[_\s]*age(?:[_\s]*days)?|customer[_\s]*tenure)\s*[:=\-]\s*(\d{1,5})\b",
        text,
    )
    if m_age:
        extra["account_age_days"] = int(m_age.group(1))

    # transaction_count_24h (e.g. "Transactions (24h): 4", "transaction_count_24h: 12")
    m_cnt = re.search(
        r"(?i)\b(?:transaction[_\s]*count[_\s]*24h|transactions?\s*\(?24h\)?|24h\s*transactions?|daily\s*txn\s*count)\s*[:=\-]\s*(\d{1,4})\b",
        text,
    )
    if m_cnt:
        extra["transaction_count_24h"] = int(m_cnt.group(1))

    # avg_transaction_amount (e.g. "Avg Transaction Amount: ₹150.00", "avg_transaction_amount: 200")
    m_avg = re.search(
        r"(?i)\b(?:avg[_\s]*transaction[_\s]*amount|average[_\s]*transaction(?:[_\s]*amount)?|avg[_\s]*spend|avg[_\s]*amount)\s*[:=\-]\s*(?:₹|rs\.?|inr|\$)?\s*([0-9]{1,3}(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)",
        text,
    )
    if m_avg:
        val = _parse_float(m_avg.group(1))
        if val is not None and val > 0:
            extra["avg_transaction_amount"] = round(val, 2)

    return extra


def parse_receipt_text(text: str) -> Dict[str, Any]:
    """
    Parse unstructured receipt/bill text into a dictionary of extracted model features.
    Only returns features that were actually detected in the receipt text.
    """
    extracted: Dict[str, Any] = {}
    if not text or not text.strip():
        return extracted

    amount = _extract_amount_from_text(text)
    if amount is not None and amount > 0:
        extracted["transaction_amount"] = amount

    txn_time = _extract_time_from_text(text)
    if txn_time is not None and 0 <= txn_time <= 86399:
        extracted["transaction_time"] = txn_time

    location = _extract_location_from_text(text)
    if location is not None:
        extracted["location"] = location

    merchant_cat = _extract_merchant_category_from_text(text)
    if merchant_cat is not None:
        extracted["merchant_category"] = merchant_cat

    extra_fields = _extract_optional_banking_fields(text)
    extracted.update(extra_fields)

    return extracted


def extract_and_impute_receipt(
    file_bytes: Optional[bytes] = None,
    filename: str = "",
    content_type: str = "",
    raw_text: Optional[str] = None,
    form_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    End-to-end receipt feature extractor and general feature imputer:
    1. Extracts all available input features from the uploaded receipt file (or raw_text).
    2. For any missing input features, assigns general baseline features from
       `GENERAL_FEATURE_DEFAULTS` (or explicit `form_overrides` when provided).
    3. Returns:
       - `final_features`: Complete 8-feature dict ready for the MLP model
       - `extracted_features`: Features fetched directly from the bill/receipt
       - `imputed_features`: Missing features filled with general baseline values
       - `missing_features`: List of feature names that were missing on the receipt
       - `raw_text`: Extracted text from the receipt
    """
    extracted: Dict[str, Any] = {}
    combined_text = raw_text or ""

    if file_bytes:
        file_text, structured_fields = extract_text_from_file(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )
        if file_text:
            combined_text = f"{combined_text}\n{file_text}".strip()
        extracted.update(structured_fields)

    if combined_text:
        text_features = parse_receipt_text(combined_text)
        for k, v in text_features.items():
            if k not in extracted or extracted[k] is None:
                extracted[k] = v

    if filename:
        fname_clean = re.sub(r"[_\-\.]+", " ", filename)
        if "location" not in extracted:
            loc_hint = _extract_location_from_text(fname_clean)
            if loc_hint:
                extracted["location"] = loc_hint
        if "merchant_category" not in extracted:
            cat_hint = _extract_merchant_category_from_text(fname_clean)
            if cat_hint:
                extracted["merchant_category"] = cat_hint

    final_features: Dict[str, Any] = {}
    imputed_features: Dict[str, Any] = {}
    missing_features: List[str] = []

    overrides = form_overrides or {}

    for key, general_default in GENERAL_FEATURE_DEFAULTS.items():
        if key in extracted and extracted[key] is not None:
            final_features[key] = extracted[key]
        else:
            missing_features.append(key)
            override_val = overrides.get(key)
            if override_val is not None and override_val != "":
                final_features[key] = override_val
                imputed_features[key] = override_val
            else:
                final_features[key] = general_default
                imputed_features[key] = general_default

    return {
        "final_features": final_features,
        "extracted_features": extracted,
        "imputed_features": imputed_features,
        "missing_features": missing_features,
        "raw_text": combined_text,
    }