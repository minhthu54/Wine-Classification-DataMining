import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Tuần 1 — Wine Dataset\nThành Viên 1: Chuyên gia Dữ liệu & Trực quan hóa"),

    nbf.v4.new_code_cell(
"import pandas as pd\n"
"import numpy as np\n"
"import matplotlib.pyplot as plt\n"
"import pickle\n"
"from sklearn.preprocessing import StandardScaler\n"
"from sklearn.decomposition import PCA"
    ),

    nbf.v4.new_markdown_cell("## Bước 1: Đọc file và đặt tên cột"),

    nbf.v4.new_code_cell(
"df = pd.read_excel('wine.data.xlsx', sheet_name='wine.data', header=0)\n"
"col_names = [\n"
"    'Class','Alcohol','Malic_Acid','Ash','Alcalinity_of_Ash',\n"
"    'Magnesium','Total_Phenols','Flavanoids','Nonflavanoid_Phenols',\n"
"    'Proanthocyanins','Color_Intensity','Hue','OD280_OD315','Proline'\n"
"]\n"
"df.columns = col_names\n"
"df['Class'] = df['Class'].astype(int)\n"
"df.head()"
    ),

    nbf.v4.new_markdown_cell("## Bước 2: Kiểm tra tổng quan dataset"),

    nbf.v4.new_code_cell(
"print(f'Kich thuoc: {df.shape[0]} mau x {df.shape[1]} cot')\n"
"print('\\nKieu du lieu:')\n"
"print(df.dtypes)\n"
"print('\\nSo mau theo nhan:')\n"
"print(df['Class'].value_counts().sort_index())"
    ),

    nbf.v4.new_markdown_cell("## Bước 3: Thống kê Max / Min / Mean"),

    nbf.v4.new_code_cell(
"features = col_names[1:]\n"
"stats = df[features].agg(['max', 'min', 'mean']).T.round(4)\n"
"stats.columns = ['Max', 'Min', 'Mean']\n"
"print(stats)"
    ),

    nbf.v4.new_markdown_cell("## Bước 4: Chuẩn hóa dữ liệu với StandardScaler"),

    nbf.v4.new_code_cell(
"scaler = StandardScaler()\n"
"X = df[features].values\n"
"X_scaled = scaler.fit_transform(X)\n"
"with open('scaler.pkl', 'wb') as f:\n"
"    pickle.dump(scaler, f)\n"
"df_scaled = pd.DataFrame(X_scaled, columns=features)\n"
"df_scaled.insert(0, 'Class', df['Class'].values)\n"
"df_scaled.to_csv('wine_scaled.csv', index=False)\n"
"print('Da luu: scaler.pkl')\n"
"print('Da luu: wine_scaled.csv')"
    ),

    nbf.v4.new_markdown_cell("## Bước 5: PCA 2D và Scatter Plot"),

    nbf.v4.new_code_cell(
"pca = PCA(n_components=2, random_state=42)\n"
"X_pca = pca.fit_transform(X_scaled)\n"
"ev = pca.explained_variance_ratio_\n"
"print(f'PC1: {ev[0]*100:.1f}%  PC2: {ev[1]*100:.1f}%  Tong: {sum(ev)*100:.1f}%')\n"
"colors  = {1: '#4C9BE8', 2: '#2DB37A', 3: '#E8613C'}\n"
"markers = {1: 'o', 2: '^', 3: 's'}\n"
"labels  = df['Class'].values\n"
"counts  = df['Class'].value_counts().sort_index()\n"
"fig, ax = plt.subplots(figsize=(9, 6.5))\n"
"fig.patch.set_facecolor('white')\n"
"ax.set_facecolor('white')\n"
"for cls in [1, 2, 3]:\n"
"    mask = labels == cls\n"
"    ax.scatter(X_pca[mask,0], X_pca[mask,1], c=colors[cls], marker=markers[cls],\n"
"               label=f'Class {cls} (n={counts[cls]})', alpha=0.85,\n"
"               edgecolors='white', linewidths=0.4, s=65, zorder=3)\n"
"ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}%)', fontsize=11)\n"
"ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}%)', fontsize=11)\n"
"ax.set_title(f'PCA 2D SCATTER PLOT - 13 CHIEU -> 2 CHIEU ({sum(ev)*100:.1f}% PHUONG SAI)', fontsize=11, fontweight='bold')\n"
"ax.grid(True, color='#e5e5e5', linewidth=0.8, zorder=0)\n"
"ax.set_axisbelow(True)\n"
"ax.legend(frameon=True, fontsize=10, loc='upper left')\n"
"plt.tight_layout()\n"
"plt.savefig('pca_scatter.png', dpi=150, bbox_inches='tight')\n"
"plt.show()"
    ),
]

nb.cells = cells
with open('wine_week1.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Tao xong: wine_week1.ipynb")