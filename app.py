import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mysql.connector
from mysql.connector import Error
import ast

# --- إعداد صفحة التطبيق ---
st.set_page_config(page_title="📦 نظام التوصية الذكي", layout="centered")

st.markdown("""
    <h1 style="text-align: center; color: #2C3E50; margin-bottom: 30px;">
        📦 نظام التوصية الذكي للمنتجات
    </h1>
""", unsafe_allow_html=True)

# --- تحسين تصميم الأزرار ---
st.markdown("""
    <style>
        .stButton > button {
            background-color: #3498DB;
            color: white;
            padding: 10px 24px;
            border: none;
            border-radius: 6px;
            margin: 5px 0;
            font-weight: bold;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #2980B9;
            color: white;
        }
        .stSlider > div {
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- تحميل البيانات من قاعدة البيانات ---
@st.cache_data(ttl=600)
def load_data():
    try:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="",
            database="recommender_system1",
            port=3306
        )

        products_df = pd.read_sql("SELECT * FROM products_clusters", conn)

        rules_df = pd.read_sql("SELECT * FROM association_rules", conn)
        # تحويل النصوص المخزنة في قواعد الارتباط إلى مجموعات فعلية
        rules_df['Antecedents'] = rules_df['Antecedents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else set())
        rules_df['Consequents'] = rules_df['Consequents'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else set())

        conn.close()
        return products_df, rules_df

    except Error as e:
        st.error(f"❌ خطأ في الاتصال بقاعدة البيانات: {e}")
        st.stop()

products_df, rules_df = load_data()

st.success("✅ تم تحميل بيانات المنتجات وقواعد الارتباط من قاعدة البيانات.")

# --- دالة التوصية ---
def recommend_combined(product_id, products_df, rules_df, n=5, cluster_col='Cluster_With_Price'):
    cluster_id = products_df.loc[products_df['ProductID'] == product_id, cluster_col].values[0]

    cluster_recs = products_df[
        (products_df[cluster_col] == cluster_id) &
        (products_df['ProductID'] != product_id)
    ]['ProductID'].tolist()

    assoc_recs = []
    relevant_rules = rules_df[rules_df['Antecedents'].apply(lambda x: isinstance(x, (set, list)) and product_id in x)]
    for consequents in relevant_rules['Consequents']:
        assoc_recs.extend(list(consequents))

    combined = cluster_recs + assoc_recs
    combined = list(dict.fromkeys(combined))
    combined = [pid for pid in combined if pid != product_id]

    return cluster_id, combined[:n]

# --- إدارة حالة الجلسة ---
if 'show_recommendations' not in st.session_state:
    st.session_state['show_recommendations'] = False
if 'ratings' not in st.session_state:
    st.session_state['ratings'] = {}
if 'selected_product' not in st.session_state:
    st.session_state['selected_product'] = products_df['ProductID'].iloc[0]
if 'cluster_method' not in st.session_state:
    st.session_state['cluster_method'] = 'Cluster_With_Price'

# --- واجهة اختيار المنتج وطريقة العنقدة ---
with st.form("selection_form"):
    st.subheader("📋 اختر رقم المنتج:")
    selected_product = st.selectbox("", products_df['ProductID'].unique(), index=products_df['ProductID'].tolist().index(st.session_state['selected_product']))
    st.session_state['selected_product'] = selected_product

    st.subheader("🎛️ اختر طريقة العنقدة:")
    cluster_method = st.selectbox("", ['Cluster_With_Price', 'Cluster_Without_Price'], index=0 if st.session_state['cluster_method']=='Cluster_With_Price' else 1)
    st.session_state['cluster_method'] = cluster_method

    submitted = st.form_submit_button("عرض التوصيات")
    reset = st.form_submit_button("إعادة تعيين")

if submitted:
    st.session_state['show_recommendations'] = True
    st.session_state['ratings'] = {}
elif reset:
    st.session_state['show_recommendations'] = False
    st.session_state['ratings'] = {}

# --- عرض التوصيات ---
if st.session_state['show_recommendations']:
    product_id = st.session_state['selected_product']
    selected_row = products_df[products_df['ProductID'] == product_id].iloc[0]
    product_name = selected_row['ProductName']

    cluster_id, recommended = recommend_combined(
        product_id,
        products_df,
        rules_df,
        n=5,
        cluster_col=st.session_state['cluster_method']
    )

    st.markdown(f"### 🛒 المنتج المحدد: **{product_name}** (ID: {product_id})")
    st.markdown(f"**📊 ينتمي للعنقود:** `{cluster_id}`")

    if recommended:
        st.markdown("### ✅ المنتجات المقترحة:")
        for pid in recommended:
            product = products_df[products_df['ProductID'] == pid].iloc[0]
            price = product.get('Price', None)
            price_display = f"💵 السعر: {price}" if pd.notna(price) else "💵 السعر: غير متوفر"
            st.write(f"🔹 **{product['ProductName']}** | {price_display}")

        st.markdown(f"**عدد المنتجات المقترحة:** `{len(recommended)}`")

        st.markdown("### 📝 قيم جودة التشابه للمنتجات المقترحة:")
        for pid in recommended:
            product = products_df[products_df['ProductID'] == pid].iloc[0]
            key = f"rating_{pid}"
            value = st.session_state['ratings'].get(key, 50)
            rating = st.slider(
                f"تقييم جودة التشابه للمنتج: {product['ProductName']} (ID: {pid})",
                min_value=0, max_value=100,
                value=value,
                key=key
            )
            st.session_state['ratings'][key] = rating

        if st.button("حفظ التقييمات"):
            try:
                conn = mysql.connector.connect(
                    host="127.0.0.1",
                    user="root",
                    password="",
                    database="recommender_system1",
                    port=3306
                )
                if conn.is_connected():
                    cursor = conn.cursor()
                    insert_sql = """
                    INSERT INTO user_similarity_ratings (ProductID, ProductName, Rating, Cluster_Method)
                    VALUES (%s, %s, %s, %s)
                    """
                    for pid in recommended:
                        key = f"rating_{pid}"
                        values = (
                            int(pid),
                            products_df.loc[products_df['ProductID'] == pid, 'ProductName'].values[0],
                            int(st.session_state['ratings'][key]),
                            st.session_state['cluster_method']
                        )
                        cursor.execute(insert_sql, values)
                    conn.commit()
                    st.success("✅ تم حفظ التقييمات في قاعدة البيانات.")
            except Error as e:
                st.error(f"❌ خطأ أثناء الحفظ في قاعدة البيانات: {e}")
            finally:
                if 'conn' in locals() and conn.is_connected():
                    cursor.close()
                    conn.close()
                    st.info("🔒 تم إغلاق الاتصال بقاعدة البيانات.")
    else:
        st.warning("⚠️ لا توجد توصيات لهذا المنتج.")

    # زر مقارنة متوسطات التقييم
    if st.button("📊 مقارنة متوسط التقييم بين الطريقتين"):
        try:
            conn = mysql.connector.connect(
                host="127.0.0.1",
                user="root",
                password="",
                database="recommender_system1",
                port=3306
            )
            if conn.is_connected():
                query = "SELECT Cluster_Method, AVG(Rating) as avg_rating FROM user_similarity_ratings GROUP BY Cluster_Method"
                df_avg = pd.read_sql(query, conn)

                st.write("### 📈 متوسطات التقييم:")
                st.dataframe(df_avg.style.format({"avg_rating": "{:.2f}"}))

                plt.figure(figsize=(6,4))
                sns.barplot(x=df_avg['Cluster_Method'], y=df_avg['avg_rating'], palette='pastel')
                plt.title("مقارنة متوسط التقييم بين الطريقتين")
                plt.ylabel("متوسط التقييم")
                plt.xlabel("طريقة العنقدة")
                st.pyplot(plt.gcf())

        except Error as e:
            st.error(f"❌ خطأ أثناء الاتصال أو القراءة من قاعدة البيانات: {e}")
        finally:
            if 'conn' in locals() and conn.is_connected():
                conn.close()
else:
    st.info("اضغط على زر 'عرض التوصيات' لعرض المنتجات المقترحة والتقييمات.")
