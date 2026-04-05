"""
seed_database.py
----------------
Creates and populates the SQLite database used by the SQL-retrieval arm
of the hybrid-RAG pipeline.

Run once (or re-run to reset):
    python seed_database.py

Tables created
--------------
  products  – catalogue of items the business sells
  orders    – individual purchase transactions
  customers – registered buyers with lifetime spend
"""

import sys
from pathlib import Path

# ── Make sure src/ is importable when run from project root ──────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import DB_PATH
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Date,
    ForeignKey,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session
from datetime import date, timedelta
import random

# ── SQLAlchemy ORM base ───────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    id       = Column(Integer, primary_key=True)
    name     = Column(String(120), nullable=False)
    category = Column(String(60),  nullable=False)
    price    = Column(Float,       nullable=False)
    stock    = Column(Integer,     nullable=False)


class Order(Base):
    __tablename__ = "orders"

    id         = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity   = Column(Integer, nullable=False)
    date       = Column(Date,    nullable=False)
    revenue    = Column(Float,   nullable=False)


class Customer(Base):
    __tablename__ = "customers"

    id          = Column(Integer, primary_key=True)
    name        = Column(String(100), nullable=False)
    region      = Column(String(60),  nullable=False)
    total_spent = Column(Float,       nullable=False)


# ── Seed data ─────────────────────────────────────────────────────────────────
PRODUCTS: list[dict] = [
    {"name": "Wireless Noise-Cancelling Headphones", "category": "Electronics",   "price": 249.99,  "stock": 142},
    {"name": "Ergonomic Mesh Office Chair",          "category": "Furniture",     "price": 389.00,  "stock": 55},
    {"name": "Stainless Steel Water Bottle (32oz)",  "category": "Outdoors",      "price": 34.95,   "stock": 310},
    {"name": "Mechanical Keyboard – TKL",            "category": "Electronics",   "price": 129.00,  "stock": 88},
    {"name": "Standing Desk Converter",              "category": "Furniture",     "price": 215.50,  "stock": 37},
    {"name": "Yoga Mat – 6mm",                       "category": "Sports",        "price": 45.00,   "stock": 200},
    {"name": "Smart LED Desk Lamp",                  "category": "Electronics",   "price": 59.99,   "stock": 175},
    {"name": "Insulated Travel Mug (16oz)",          "category": "Kitchen",       "price": 24.95,   "stock": 420},
    {"name": "Portable Bluetooth Speaker",           "category": "Electronics",   "price": 89.00,   "stock": 95},
    {"name": "Bamboo Cutting Board Set",             "category": "Kitchen",       "price": 38.50,   "stock": 260},
    {"name": "Running Shoes – Men's",                "category": "Sports",        "price": 119.95,  "stock": 130},
    {"name": "Running Shoes – Women's",              "category": "Sports",        "price": 114.95,  "stock": 145},
    {"name": "Resistance Band Set (5 levels)",       "category": "Sports",        "price": 27.99,   "stock": 350},
    {"name": "Non-Stick Cookware Set (10 pcs)",      "category": "Kitchen",       "price": 179.00,  "stock": 60},
    {"name": "UV-Protection Sunglasses",             "category": "Accessories",   "price": 65.00,   "stock": 190},
    {"name": "Leather Wallet – Bifold",              "category": "Accessories",   "price": 42.00,   "stock": 220},
    {"name": "USB-C Hub (7-in-1)",                   "category": "Electronics",   "price": 54.99,   "stock": 300},
    {"name": "Cast Iron Skillet (12-inch)",          "category": "Kitchen",       "price": 49.95,   "stock": 115},
    {"name": "Hiking Backpack 40L",                  "category": "Outdoors",      "price": 159.00,  "stock": 72},
    {"name": "Foam Roller – High-Density",           "category": "Sports",        "price": 32.00,   "stock": 240},
]

CUSTOMERS: list[dict] = [
    {"name": "Alice Nguyen",     "region": "North America", "total_spent": 1254.75},
    {"name": "Bob Thornton",     "region": "Europe",        "total_spent": 876.20},
    {"name": "Carol Santos",     "region": "South America", "total_spent": 2103.50},
    {"name": "David Kim",        "region": "Asia Pacific",  "total_spent": 589.00},
    {"name": "Eva Fischer",      "region": "Europe",        "total_spent": 3421.80},
    {"name": "Frank Osei",       "region": "Africa",        "total_spent": 412.30},
    {"name": "Grace Liu",        "region": "Asia Pacific",  "total_spent": 1780.60},
    {"name": "Henry Martínez",   "region": "North America", "total_spent": 945.15},
    {"name": "Isabel Costa",     "region": "Europe",        "total_spent": 2259.90},
    {"name": "James Patel",      "region": "Asia Pacific",  "total_spent": 330.40},
    {"name": "Karen O'Brien",    "region": "North America", "total_spent": 1125.00},
    {"name": "Liam Johansson",   "region": "Europe",        "total_spent": 678.55},
    {"name": "Mia Chen",         "region": "Asia Pacific",  "total_spent": 4050.25},
    {"name": "Noah Williams",    "region": "North America", "total_spent": 2815.70},
    {"name": "Olivia Dupont",    "region": "Europe",        "total_spent": 719.90},
    {"name": "Paulo Ferreira",   "region": "South America", "total_spent": 1533.45},
    {"name": "Quinn Nakamura",   "region": "Asia Pacific",  "total_spent": 987.00},
    {"name": "Rachel Adeyemi",   "region": "Africa",        "total_spent": 265.80},
    {"name": "Samuel Grant",     "region": "North America", "total_spent": 3677.15},
    {"name": "Tina Müller",      "region": "Europe",        "total_spent": 1492.60},
]


def build_orders(products: list[Product], n: int = 20) -> list[dict]:
    """Generate n realistic orders spread across the last 12 months."""
    random.seed(42)
    base_date = date(2025, 4, 1)
    orders = []
    for i in range(n):
        product  = products[i % len(products)]
        qty      = random.randint(1, 10)
        order_dt = base_date + timedelta(days=random.randint(0, 365))
        orders.append({
            "product_id": product.id,
            "quantity":   qty,
            "date":       order_dt,
            "revenue":    round(product.price * qty, 2),
        })
    return orders


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Ensure the database directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

    # Drop and recreate all tables (idempotent re-seed)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # ── products ──────────────────────────────────────────────────────────
        product_objs = [Product(**p) for p in PRODUCTS]
        session.add_all(product_objs)
        session.flush()   # populate auto-increment IDs before referencing them

        # ── orders ────────────────────────────────────────────────────────────
        order_rows = build_orders(product_objs)
        session.add_all([Order(**o) for o in order_rows])

        # ── customers ─────────────────────────────────────────────────────────
        session.add_all([Customer(**c) for c in CUSTOMERS])

        session.commit()

    # Quick sanity-check
    with engine.connect() as conn:
        for table in ("products", "orders", "customers"):
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"  ✓  {table:<12} — {count} rows")

    print(f"\nDatabase created at: {DB_PATH.resolve()}")


if __name__ == "__main__":
    main()
