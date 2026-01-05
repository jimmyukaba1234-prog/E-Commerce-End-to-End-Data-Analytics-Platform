# ========================================================
# E-COMMERCE DASHBOARD (All analyses, multi-tab)
# ========================================================

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
import dash_bootstrap_components as dbc

# -------------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------------
conn = sqlite3.connect("Ecommerce.db")
print("Connected to Ecommerce.db")
# ---------------- Load tables into DataFrames ----------------
df_customers = pd.read_sql("SELECT * FROM customers_clean", conn)
df_products  = pd.read_sql("SELECT * FROM products_clean", conn)
df_orders    = pd.read_sql("SELECT * FROM orders_clean", conn)
df_reviews   = pd.read_sql("SELECT * FROM reviews_clean", conn)

# ---------------- Define active/inactive statuses ----------------
active_statuses = ["active", "gold", "silver", "premium"]
inactive_statuses = ["inactive", "suspended"]

# ---------------- Compute metrics ----------------
total_customers = df_customers["customer_id"].nunique()
active_customers = df_customers.loc[
    df_customers["customer_status"].str.lower().isin(active_statuses), "customer_id"
].nunique()
#inactive_customers = df_customers.loc[
    #df_customers["customer_status"].str.lower().isin(inactive_statuses), "customer_id"].nunique()
total_orders = df_orders["order_id"].nunique()
total_revenue = round(df_orders["total_amount_usd"].sum(), 2)

# ---------------- Create KBI DataFrame ----------------
kbi = pd.DataFrame({
    "Metric": [
        "Total Customers",
        "Active Customers",
        #"Inactive Customers",
        "Total Orders",
        "Total Revenue (USD)"
    ],
    "Value": [
        total_customers,
        active_customers,
        #inactive_customers,
        total_orders,
        total_revenue
    ]
})


# ========================================================
# ANALYSIS 1 — DELIVERY TIME vs PRODUCT RATING
# ========================================================
query_delivery = """
SELECT p.product_id, p.rating AS product_rating,
julianday(o.actual_delivery) - julianday(o.order_date_date) AS delivery_time_days
FROM products_clean p
JOIN orders_clean o ON p.product_id=o.product_id
WHERE p.rating IS NOT NULL AND o.actual_delivery IS NOT NULL AND o.order_date_date IS NOT NULL
"""
df_delivery = pd.read_sql(query_delivery, conn)

fig_delivery = px.scatter(
    df_delivery,
    x="delivery_time_days",
    y="product_rating",
    trendline="ols",
    title="Delivery Time vs Product Rating",
    labels={"delivery_time_days": "Delivery Time (Days)", "product_rating": "Product Rating"}
)

# ========================================================
# ANALYSIS 2 — DELIVERY RATE BY WAREHOUSE
# ========================================================
query_warehouse = """
SELECT warehouse_id, 
COUNT(actual_delivery) AS shipped_orders, 
COUNT(DISTINCT order_id) AS total_orders, 
ROUND(100.0*COUNT(actual_delivery)/COUNT(DISTINCT order_id),2) AS delivery_rate
FROM orders_clean
GROUP BY warehouse_id
"""
df_warehouse = pd.read_sql(query_warehouse, conn)

fig_warehouse = px.bar(
    df_warehouse,
    x="warehouse_id",
    y="delivery_rate",
    title="Delivery Rate by Warehouse",
    labels={"warehouse_id":"Warehouse ID","delivery_rate":"Delivery Rate (%)"}
)
fig_warehouse.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
fig_warehouse.update_layout(yaxis_ticksuffix="%", xaxis_tickangle=-30)

# ========================================================
# ANALYSIS 3 — FAILED PAYMENT RATE
# ========================================================
query_payment = payments = """
WITH normalized_orders AS (
    SELECT
        order_id,
        payment_method,
        LOWER(TRIM(CAST(payment_status AS TEXT))) AS payment_status
    FROM orders_clean
),
payment_failure_stats AS (
    SELECT
        payment_method,
        COUNT(order_id) AS all_payments,
        SUM(
            CASE 
                WHEN payment_status = 'failed' THEN 1 
                ELSE 0 
            END
        ) AS failed_payments
    FROM normalized_orders
    GROUP BY payment_method
)
SELECT
    payment_method,
    all_payments,
    failed_payments,
    ROUND(
        100.0 * failed_payments / NULLIF(all_payments, 0),
        2
    ) AS failed_payment_rate
FROM payment_failure_stats
WHERE all_payments > 0
ORDER BY failed_payment_rate DESC;"""

df_payment = pd.read_sql(query_payment, conn)

fig_payment = px.bar(
    df_payment,
    x="payment_method",
    y="failed_payment_rate",
    title="Failed Payment Rate by Payment Method",
    labels={"payment_method":"Payment Method","failed_payment_rate":"Failed Payment Rate (%)"}
)
fig_payment.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
fig_payment.update_layout(yaxis_ticksuffix="%", xaxis_tickangle=-30)

# ========================================================
# ANALYSIS 4 — MONTHLY SALES, PROFIT, ORDERS BY YEAR
# ========================================================
query_sales = """
SELECT 
strftime('%Y', o.order_date_date) AS order_year,
strftime('%m', o.order_date_date) AS order_month,
ROUND(SUM(
    CASE o.currency
        WHEN 'USD' THEN o.total_amount
        WHEN 'EUR' THEN o.total_amount * 1.09
        WHEN 'GBP' THEN o.total_amount * 1.27
        ELSE 0 END), 2) AS sales_usd,
COUNT(o.order_id) AS total_orders,
ROUND(SUM(
    CASE o.currency
        WHEN 'USD' THEN (p.price - p.cost)
        WHEN 'EUR' THEN (p.price - p.cost) * 1.09
        WHEN 'GBP' THEN (p.price - p.cost) * 1.27
        ELSE 0 END),2) AS profit_usd
FROM orders_clean o
JOIN products_clean p ON o.product_id = p.product_id
WHERE o.order_date_date IS NOT NULL
GROUP BY order_year, order_month
ORDER BY order_year, order_month
"""
df_sales = pd.read_sql(query_sales, conn)
df_sales["order_year"] = df_sales["order_year"].astype(int)
df_sales["order_month"] = df_sales["order_month"].astype(int)
df_sales["month"] = pd.to_datetime(df_sales["order_month"], format="%m").dt.strftime("%b")
years = sorted(df_sales["order_year"].unique())

def create_year_dropdown_plot(metric, title):
    fig = go.Figure()
    for i, year in enumerate(years):
        df_year = df_sales[df_sales["order_year"]==year]
        fig.add_trace(go.Scatter(
            x=df_year["month"],
            y=df_year[metric],
            mode="lines+markers",
            name=str(year),
            visible=True if i==0 else False
        ))
    fig.update_layout(
        updatemenus=[{
            "buttons":[
                {"label":str(year),
                 "method":"update",
                 "args":[{"visible":[y==year for y in years]},{"title":f"{title} – {year}"}]
                } for year in years
            ],
            "direction":"down","x":0.02,"y":1.15
        }],
        title=f"{title} – {years[0]}",
        xaxis_title="Month",
        yaxis_title=metric
    )
    return fig

fig_sales_trend = create_year_dropdown_plot("sales_usd","Monthly Sales Trend")
fig_profit_trend = create_year_dropdown_plot("profit_usd","Monthly Profit Trend")
fig_orders_trend = create_year_dropdown_plot("total_orders","Monthly Orders Trend")

# ========================================================
# ANALYSIS 5 — CURRENCY SALES & PROFIT
# ========================================================
query_currency = """
SELECT o.currency,
ROUND(SUM(CASE 
    WHEN o.currency='USD' THEN o.total_amount
    WHEN o.currency='EUR' THEN o.total_amount*1.09
    WHEN o.currency='GBP' THEN o.total_amount*1.27
    ELSE 0 END), 2) AS total_sales_usd,
ROUND(SUM(CASE 
    WHEN o.currency='USD' THEN (p.price-p.cost)
    WHEN o.currency='EUR' THEN (p.price-p.cost)*1.09
    WHEN o.currency='GBP' THEN (p.price-p.cost)*1.27
    ELSE 0 END), 2) AS total_profit_usd
FROM orders_clean o
JOIN products_clean p ON o.product_id=p.product_id
GROUP BY o.currency
ORDER BY total_profit_usd DESC
"""
df_currency = pd.read_sql(query_currency, conn)

fig_sales_currency = px.pie(df_currency, names="currency", values="total_sales_usd", title="Sales Contribution by Currency", hole=0.4)
fig_profit_currency = px.pie(df_currency, names="currency", values="total_profit_usd", title="Profit Contribution by Currency", hole=0.4)

# ========================================================
# ANALYSIS 6 — CUSTOMER SPENDING TIERS
# ========================================================
query_tiers = """
SELECT
    CASE 
        WHEN total_spent>10000 THEN 'VIP (>10k)'
        WHEN total_spent>5000 THEN 'Premium (5k-10k)'
        WHEN total_spent>1000 THEN 'Regular (1k-5k)'
        WHEN total_spent>100 THEN 'Occasional (100-1k)'
        ELSE 'Low (<100)' END AS spending_tier,
    COUNT(*) AS customer_count,
    ROUND(AVG(total_spent),2) AS avg_spent,
    ROUND(AVG(account_age_days),1) AS avg_account_age
FROM customers
WHERE total_spent IS NOT NULL
GROUP BY spending_tier
ORDER BY avg_spent DESC
"""
df_tiers = pd.read_sql(query_tiers, conn)

fig_tiers = px.pie(
    df_tiers, names="spending_tier", values="customer_count",
    hover_data=["avg_spent","avg_account_age"],
    title="Customer Distribution by Spending Tier"
)
fig_tiers.update_traces(textinfo="percent+label")

# ========================================================
# ANALYSIS 7 — CUSTOMER RETENTION & CHURN
# ========================================================
monthly_active = pd.read_sql_query("""
SELECT strftime('%Y-%m', order_date_date) AS year_month,
COUNT(DISTINCT customer_id) AS active_customers
FROM orders_clean
GROUP BY 1 ORDER BY 1
""", conn)

fig_active = px.line(
    monthly_active, x="year_month", y="active_customers",
    title="Monthly Active Customers Over Time",
    labels={"year_month":"Year-Month","active_customers":"Active Customers"},
    markers=True
)
fig_active.update_yaxes(range=[0,5000])

monthly_churn_rate = pd.read_sql_query("""
WITH monthly_customers AS (
    SELECT DISTINCT customer_id,strftime('%Y-%m',order_date_date) AS year_month
    FROM orders_clean
),
churn_base AS (
    SELECT m1.year_month AS month,
    COUNT(DISTINCT m1.customer_id) AS active_customers,
    COUNT(DISTINCT CASE WHEN m2.customer_id IS NULL THEN m1.customer_id END) AS churned_customers
    FROM monthly_customers m1
    LEFT JOIN monthly_customers m2
    ON m1.customer_id=m2.customer_id AND m2.year_month=strftime('%Y-%m',date(m1.year_month||'-01','+1 month'))
    GROUP BY m1.year_month
)
SELECT month, active_customers, churned_customers,
ROUND(1.0*churned_customers/active_customers,4) AS churn_rate
FROM churn_base ORDER BY month
""", conn)

fig_churn_bar = px.bar(
    monthly_churn_rate, x="month", y=["active_customers","churned_customers"],
    barmode="group", title="Active vs Churned Customers per Month",
    labels={"value":"Number of Customers","month":"Month","variable":"Customers"},
    color_discrete_map={"active_customers":"green","churned_customers":"red"},
    height=500
)
fig_churn_bar.update_yaxes(range=[0,4000])

# ========================================================
# ANALYSIS 8 — CUSTOMER ATTRIBUTES AND CHURN
# ========================================================
churn_attr = pd.read_sql_query("""
WITH monthly_customers AS (
    SELECT DISTINCT customer_id,strftime('%Y-%m',order_date_date) AS year_month
    FROM orders_clean
),
churn_flag AS (
    SELECT m1.customer_id,m1.year_month,
    CASE WHEN m2.customer_id IS NULL THEN 1 ELSE 0 END AS churned
    FROM monthly_customers m1
    LEFT JOIN monthly_customers m2
    ON m1.customer_id=m2.customer_id AND m2.year_month=strftime('%Y-%m',date(m1.year_month||'-01','+1 month'))
)
SELECT c.customer_id,c.country,c.credit_tier,c.loyalty_score,
date(c.date_of_birth) AS date_of_birth,
CAST((julianday('now')-julianday(c.date_of_birth))/365.25 AS INT) AS age,
c.newsletter_subscribed,c.marketing_consent,cf.churned
FROM churn_flag cf
JOIN customers_clean c ON cf.customer_id=c.customer_id
LIMIT 100
""", conn)

fig_churn_attr = px.scatter(
    churn_attr,
    x="loyalty_score", y="age", size="churned",
    color="credit_tier", symbol="newsletter_subscribed",
    hover_name="customer_id", title="Customer Churn by Loyalty Score and Age",
    size_max=20,
    labels={"loyalty_score":"Loyalty Score","age":"Age","churned":"Churned","newsletter_subscribed":"Newsletter"}
)


churn_flag = """WITH monthly_customers AS (
    SELECT DISTINCT
        customer_id,
        strftime('%Y-%m', order_date_date) AS year_month
    FROM orders_clean
),
churn_flag AS (
    SELECT 
        m1.customer_id,
        m1.year_month,
        CASE 
            WHEN m2.customer_id IS NULL THEN 1
            ELSE 0
        END AS churned
    FROM monthly_customers m1
    LEFT JOIN monthly_customers m2
        ON m1.customer_id = m2.customer_id
       AND m2.year_month = strftime('%Y-%m', date(m1.year_month || '-01', '+1 month'))
)
SELECT *
FROM churn_flag
LIMIT 100;"""
churn_flag = pd.read_sql_query(churn_flag, conn)

fig_churn_scatter = px.scatter(
    churn_flag.assign(churned_str=churn_flag["churned"].astype(str)),
    x="customer_id",
    y="year_month",
    color="churned_str",
    title="Customer Churn Over Time",
    labels={"customer_id": "Customer ID", "year_month": "Year-Month", "churned_str": "Churned"},
    color_discrete_map={"0": "green", "1": "red"}
)


#Which country represent our highest customer concentration versus our highest revenue per capita?

Revenue_by_country = """
SELECT
country,
COUNT(customer_id) AS customer_count,
SUM(total_spent) AS revenue
FROM customers_clean
GROUP BY country
ORDER BY revenue DESC;"""
Revenue_by_country = pd.read_sql_query(Revenue_by_country, conn)


fig_map = px.choropleth(
    Revenue_by_country,
    locations="country",
    locationmode="country names",
    color="revenue",
    hover_name="country",
    color_continuous_scale="Viridis",
    title="Revenue by Country"
)

fig_map.update_layout(
    autosize=True,         # auto-adjust to container
    margin=dict(l=0, r=0, t=40, b=0),  # remove big default margins
)

# Top 10% customers by total spent
customer_rank = """
WITH top_customers AS (
    SELECT *,
           PERCENT_RANK() OVER (ORDER BY total_spent DESC) AS customer_rank
    FROM customers_clean
)
SELECT
    customer_id,
    total_spent,
    customer_rank
FROM top_customers
WHERE customer_rank >= 0.9;"""
customer_rank = pd.read_sql_query(customer_rank, conn)
customer_rank = customer_rank.sort_values(by="total_spent", ascending=False).head(10)

fig_customer_rank = px.bar(
    customer_rank,
    x="total_spent",
    y="customer_id",
    orientation="h",
    color="customer_rank",  
    text="total_spent",
    title="Top 10 Customers by Total Spent", 
    color_continuous_scale="Viridis"
)

fig_customer_rank.update_layout(yaxis={'categoryorder':'total ascending'},  
    xaxis_title="Total Spent",
    yaxis_title="Customer ID",
    height=500
)


volume_drivers = """
SELECT
    p.product_id,
    p.product_name,
    SUM(o.total_amount) AS revenue,
    SUM(p.cost * o.quantity) AS cost,
    SUM(o.total_amount) - SUM(p.cost * o.quantity) AS profit,
    (SUM(o.total_amount) - SUM(p.cost * o.quantity))/SUM(o.total_amount) AS margin
FROM orders_clean o
JOIN products p ON o.product_id = p.product_id
GROUP BY p.product_id
ORDER BY revenue DESC
LIMIT 100;"""
volume_drivers = pd.read_sql_query(volume_drivers, conn)

top_revenue = volume_drivers.sort_values("revenue", ascending=False).head(10)

fig_revenue = px.bar(
    top_revenue,
    x="revenue",
    y="product_id",
    orientation="h",
    color="margin", 
    text="profit",
    title="Top 10 Products by Revenue & Profit"
)
fig_revenue.update_layout(yaxis={'categoryorder':'total ascending'})


# Which products fall into the bottom 5% of sales and ratings, suggesting a need for inventory clearance or quality review?

bottom_products = """
WITH product_sales AS (
    SELECT
        p.product_id,
        SUM(o.total_amount) AS revenue
    FROM orders_clean o
    JOIN products_clean p 
        ON o.product_id = p.product_id
    GROUP BY p.product_id
),
product_ratings AS (
    SELECT
        product_id,
        AVG(rating) AS avg_rating
    FROM reviews_clean
    GROUP BY product_id
),
ranked_products AS (
    SELECT 
        ps.product_id,
        ps.revenue,
        pr.avg_rating,
        ROW_NUMBER() OVER (ORDER BY ps.revenue ASC, pr.avg_rating ASC) AS rn,
        COUNT(*) OVER () AS total_products
    FROM product_sales ps
    JOIN product_ratings pr 
        ON ps.product_id = pr.product_id
)
SELECT product_id, revenue, avg_rating
FROM ranked_products
WHERE rn <= MIN(100, CAST(0.05 * total_products AS INTEGER))
LIMIT 100;
"""
bottom_products = pd.read_sql_query(bottom_products, conn)


fig_rating = px.scatter(
    bottom_products,
    x="avg_rating",
    y="revenue",
    size="revenue",          
    color="avg_rating",      
    hover_name="product_id", 
    title="Revenue vs Average Rating per Product"
)

fig_rating.update_layout(
    xaxis_title="Average Rating",
    yaxis_title="Revenue"
)

# ========================================================
# DASH APP LAYOUT
# ========================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

def graph_card(figure, height=420):
    figure.update_layout(height=height)

    return dbc.Card(
        dbc.CardBody(
            dcc.Graph(
                figure=figure,
                config={"displayModeBar": False},
                style={"width": "100%", "height": f"{height}px"},
            )
        ),
        style={
            "borderRadius": "16px",
            "boxShadow": "0 8px 18px rgba(0,0,0,0.08)",
            "border": "none",
            "backgroundColor": "white",
            "marginBottom": "26px",
            "padding": "10px",
        },
    )

app.layout = dbc.Container(
    [
        dbc.Card(
            dbc.CardBody(
                html.H2(
                    "📊 E-Commerce Analytics Dashboard",
                    className="text-center",
                    style={
                        "fontWeight": "700",
                        "margin": 0,
                        "color": "white",
                    },
                ),
                style={
                    "padding": "28px",   # ← makes the blue background thicker
                },
            ),
            style={
                "backgroundColor": "#0d6efd",  # Bootstrap primary blue
                "borderRadius": "14px",
                "border": "none",
                "boxShadow": "0 4px 12px rgba(0,0,0,0.15)",
                "marginBottom": "30px",
            },
        ),
        
        # ===== KBI ROW =====
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(
                                    metric,
                                    style={
                                        "fontSize": "12px",
                                        "textTransform": "uppercase",
                                        "color": "#6c757d",
                                        "fontWeight": "600",
                                        "letterSpacing": "0.5px",
                                    },
                                ),

                                html.Div(
                                    f"{value:,.0f}" if isinstance(value, (int, float)) else value,
                                    style={
                                        "fontSize": "32px",
                                        "fontWeight": "700",
                                        "marginTop": "6px",
                                        "color": "#212529",
                                    },
                                ),
                            ]
                        ),
                        style={
                            "borderRadius": "12px",
                            "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
                            "border": "none",
                        },
                    ),
                    width=3,
                )
                for metric, value in zip(kbi["Metric"], kbi["Value"])
            ],
            className="mb-4",
        ),
        dcc.Tabs(
            [
                # ========= TAB 1 =========
                dcc.Tab(
                    label="Orders, Payments and Logistics",
                    style={"fontWeight": "bold", "fontSize": "18px"},
                    selected_style={
                        "fontWeight": "bold",
                        "fontSize": "18px",
                        "color": "#0d6efd",
                        "borderBottom": "3px solid #0d6efd",
                    },
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    graph_card(fig_payment),
                                    md=6,
                                ),
                                dbc.Col(
                                    graph_card(fig_warehouse),
                                    md=6,
                                ),
                            ],
                            className="mb-4",
                        ),

                        dbc.Row(
                            [
                                dbc.Col(
                                    graph_card(fig_delivery),
                                    md=6,
                                ),
                                dbc.Col(
                                    graph_card(fig_sales_trend),
                                    md=6,
                                ),
                            ],
                            className="mb-4",
                        ),

                        dbc.Row(
                            [
                                dbc.Col(
                                    graph_card(fig_profit_trend),
                                    md=6,
                                ),
                                dbc.Col(
                                    graph_card(fig_orders_trend),
                                    md=6,
                                ),
                            ],
                            className="mb-4",
                        ),


                        #graph_card(fig_payment),

                        #graph_card(fig_warehouse),

                        #graph_card( fig_delivery),

                        #graph_card(fig_sales_trend),

                        #graph_card(fig_profit_trend),

                        #graph_card(fig_orders_trend),

                        html.H4("Currency Insights", className="mt-3"),

                        dbc.Row(
                            [
                                dbc.Col(
                                    graph_card(fig_sales_currency),
                                    md=6,
                                ),
                                dbc.Col(
                                    graph_card(fig_profit_currency),
                                    md=6,
                                ),
                            ],
                            className="mb-4",
                        ),

                        graph_card(fig_tiers),
                    ],
                ),

                # ========= TAB 2 =========
                dcc.Tab(
                    label="Customers and Products",
                    style={"fontWeight": "bold", "fontSize": "18px"},
                    selected_style={
                        "fontWeight": "bold",
                        "fontSize": "18px",
                        "color": "#0d6efd",
                        "borderBottom": "3px solid #0d6efd",
                    },
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(
                                    graph_card(fig_active),
                                    md=6,
                                ),
                                dbc.Col(
                                    graph_card(fig_map),
                                    md=6,
                                ),
                            ],
                            className="mb-4",
                        ),
                        
                        graph_card(fig_churn_bar),
                        graph_card(fig_churn_attr),
                        graph_card(fig_churn_scatter),
                        graph_card(fig_customer_rank),
                        graph_card(fig_revenue),
                        graph_card(fig_rating),
                        
        
                    ],
                ),

            ]
        ),
    ],
    fluid=True,
)

       
# ========================================================
# RUN APP
# ========================================================
if __name__ == "__main__":
    app.run_server(debug=True)