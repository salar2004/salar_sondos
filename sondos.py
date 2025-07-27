import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. تحميل البيانات
invoices_df = pd.read_csv("Invoices_Dataset_for_Association_Rules.csv")

# 2. التحقق من القيم المفقودة
print("عدد القيم المفقودة في كل عمود:")
print(invoices_df.isnull().sum())

# 3. التحقق من التكرارات وحذفها
print(f"عدد الصفوف قبل حذف التكرارات: {invoices_df.shape[0]}")
invoices_df.drop_duplicates(inplace=True)
print(f"عدد الصفوف بعد حذف التكرارات: {invoices_df.shape[0]}")

# 4. تحويل إلى Basket Format: قائمة المنتجات لكل فاتورة
basket_df = invoices_df.groupby('InvoiceID')['ProductID'].apply(list).reset_index()
print("عرض أول 5 فواتير مع المنتجات:")
print(basket_df.head())

# 5. تجهيز قائمة معاملات للترميز
transactions = basket_df['ProductID'].tolist()

# 6. ترميز المعاملات إلى مصفوفة ثنائية
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

print("عرض أول 5 أسطر من البيانات المشفرة:")
print(df_encoded.head())

# 7. استخراج المجموعات المتكررة بدعم أدناه 5%
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
print("المجموعات المتكررة (frequent itemsets):")
print(frequent_itemsets)

# 8. استخراج قواعد الارتباط بالرفع >=1 والثقة >=0.6
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules = rules[rules['confidence'] >= 0.6]

print("أهم 10 قواعد ارتباط:")
print(rules.head(10))

# تحميل بيانات المنتجات
products_df = pd.read_csv("Extended_Products_Dataset__25_Products_ (1).csv")



# عرض أول 5 أسطر لفهم شكل البيانات
print(products_df.head())

# التحقق من القيم المفقودة في الأعمدة
print(products_df.isnull().sum())

# تعويض القيم المفقودة في ConnectivityType بـ "Unknown"
products_df['ConnectivityType'] = products_df['ConnectivityType'].fillna('Unknown')


# تأكد ما عاد فيه قيم مفقودة
print(products_df['ConnectivityType'].isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

numeric_cols = ['Price', 'Rating', 'Stock', 'WeightKg', 'VolumeCm3', 'PowerWatt', 'WarrantyYears']

# عرض Boxplot لكل عمود
for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=products_df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# حساب Z-score وتحديد القيم الشاذة (التي تتعدى 3 أو -3)
from scipy import stats

outliers = {}
for col in numeric_cols:
    z_scores = stats.zscore(products_df[col])
    outlier_indices = np.where(np.abs(z_scores) > 3)[0]
    outliers[col] = outlier_indices
    print(f"عدد القيم الشاذة في {col}: {len(outlier_indices)}")

# حساب IQR
Q1 = products_df['Price'].quantile(0.25)
Q3 = products_df['Price'].quantile(0.75)
IQR = Q3 - Q1

# حد القيم الشاذة العليا
upper_bound = Q3 + 1.5 * IQR

# عرض القيم الشاذة في Price
outliers = products_df[products_df['Price'] > upper_bound]
print("القيم الشاذة في Price:")
print(outliers[['ProductID', 'ProductName', 'Price']])

# تعويض القيم الشاذة بالوسيط
median_price = products_df['Price'].median()
products_df.loc[products_df['Price'] > upper_bound, 'Price'] = median_price

print(f"تم استبدال القيم الشاذة في Price بالوسيط: {median_price}")

# التأكد بعد التعديل
outliers_after = products_df[products_df['Price'] > upper_bound]
print(f"عدد القيم الشاذة بعد المعالجة: {outliers_after.shape[0]}")

plt.figure(figsize=(8,4))
sns.violinplot(x=products_df['Price'], color='lightgreen')
plt.title('Violin Plot of Price')
plt.show()

categorical_cols = ['Category', 'Brand', 'SupplierCountry', 'ConnectivityType', 
                    'MaterialType', 'UsageType', 'PriceCategory']

products_encoded = pd.get_dummies(products_df, columns=categorical_cols)

print(f"شكل البيانات بعد الترميز: {products_encoded.shape}")
print(products_encoded.head(3))