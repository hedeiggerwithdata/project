import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import math
from sklearn.inspection import partial_dependence
from collections import OrderedDict
from IPython.display import display
import shap
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from IPython.display import display
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import shap

SHAP_CMAP_COLORS = ("#3B4CC0", "#7B6FD0", "#B40426")   # global beeswarm과 동일 톤
SHAP_BAR_COLOR = "#7B6FD0"
SHAP_FACE = "#FCFCFD"
SHAP_SPINE = "#BFC5CE"
SHAP_GRID_X_ALPHA = 0.22
SHAP_GRID_Y_ALPHA = 0.14

def _make_shap_cmap():
    # Low -> Blue, Mid -> Purple, High -> Red
    # 컬러바 표시상 아래쪽 Low=파랑, 위쪽 High=빨강
    return LinearSegmentedColormap.from_list(
        "shap_red_purple_blue",
        ["#1E88E5", "#8E44AD", "#FF0051"]
    )
def _style_axis(ax, ygrid=True, xgrid=True):
    ax.set_facecolor(SHAP_FACE)

    if xgrid:
        ax.grid(axis="x", alpha=SHAP_GRID_X_ALPHA, linewidth=0.7)
    else:
        ax.grid(False, axis="x")

    if ygrid:
        ax.grid(axis="y", alpha=SHAP_GRID_Y_ALPHA, linewidth=0.6, linestyle=(0, (1, 3)))
    else:
        ax.grid(False, axis="y")

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SHAP_SPINE)
    ax.spines["bottom"].set_color(SHAP_SPINE)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(axis="both", labelsize=10, colors="black")


def _match_cat_base(rest_name, categorical_bases):
    if categorical_bases is None:
        return None
    for base in sorted(categorical_bases, key=len, reverse=True):
        if rest_name == base:
            return base, None
        if rest_name.startswith(base + "_"):
            return base, rest_name[len(base) + 1:]
    return None


def _build_meta(columns, categorical_bases):
    rows = []
    for col in columns:
        if col.startswith("remainder__"):
            clean = col.replace("remainder__", "")
            rows.append(
                {"encoded_col": col, "group": clean, "group_type": "numeric", "level": None}
            )
        elif col.startswith("cat__"):
            rest = col.replace("cat__", "")
            matched = _match_cat_base(rest, categorical_bases)
            if matched is not None:
                base, level = matched
            else:
                if "_" in rest:
                    base, level = rest.split("_", 1)
                else:
                    base, level = rest, None
            rows.append(
                {"encoded_col": col, "group": base, "group_type": "categorical", "level": level}
            )
        else:
            rows.append(
                {"encoded_col": col, "group": col, "group_type": "numeric", "level": None}
            )
    return pd.DataFrame(rows)


def _resolve_feature_to_group(feature, meta):
    encoded_to_group = dict(zip(meta["encoded_col"], meta["group"]))
    available_groups = set(meta["group"].tolist())

    if feature in available_groups:
        return feature
    if feature in encoded_to_group:
        return encoded_to_group[feature]
    if f"remainder__{feature}" in encoded_to_group:
        return encoded_to_group[f"remainder__{feature}"]
    raise ValueError(f"Feature '{feature}' not found.")


def _resolve_feature_to_encoded(feature, meta, X_cols=None):
    group_to_cols = meta.groupby("group")["encoded_col"].apply(list).to_dict()
    if X_cols is not None and feature in X_cols:
        return feature
    if X_cols is not None and f"remainder__{feature}" in X_cols:
        return f"remainder__{feature}"
    if feature in group_to_cols:
        return group_to_cols[feature][0]
    raise ValueError(f"Feature '{feature}' not found.")


def _pretty_name(col):
    if col.startswith("remainder__"):
        return col.replace("remainder__", "")
    if col.startswith("cat__"):
        return col.replace("cat__", "")
    return col


def _best_interaction_feature(x_vec, shap_vec, candidate_df, exclude_cols):
    best_col = None
    best_score = -np.inf
    for col in candidate_df.columns:
        if col in exclude_cols:
            continue
        z = pd.to_numeric(candidate_df[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x_vec) & np.isfinite(shap_vec) & np.isfinite(z)
        if valid.sum() < 20:
            continue
        corr1 = np.corrcoef(x_vec[valid], z[valid])[0, 1]
        corr2 = np.corrcoef(shap_vec[valid], z[valid])[0, 1]
        corr1 = 0.0 if not np.isfinite(corr1) else abs(corr1)
        corr2 = 0.0 if not np.isfinite(corr2) else abs(corr2)
        score = corr1 + corr2
        if score > best_score:
            best_score = score
            best_col = col
    return best_col


def _get_color_values_and_norm(vals, q_low=5, q_high=95):
    arr = pd.to_numeric(vals, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(arr)

    if valid.sum() == 0:
        return arr, None

    vmin, vmax = np.nanpercentile(arr[valid], [q_low, q_high])
    if np.isclose(vmin, vmax):
        center = np.nanmedian(arr[valid])
        vmin, vmax = center - 1e-9, center + 1e-9

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    return arr, norm


def _density_order(x, y, bins=40):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() == 0:
        return np.arange(len(x))

    xv = x[valid]
    yv = y[valid]

    x_bins = np.linspace(np.nanmin(xv), np.nanmax(xv), bins + 1)
    y_bins = np.linspace(np.nanmin(yv), np.nanmax(yv), bins + 1)

    x_idx = np.clip(np.digitize(xv, x_bins) - 1, 0, bins - 1)
    y_idx = np.clip(np.digitize(yv, y_bins) - 1, 0, bins - 1)

    hist = np.zeros((bins, bins), dtype=int)
    for xi, yi in zip(x_idx, y_idx):
        hist[xi, yi] += 1

    density = np.full(len(x), -1, dtype=float)
    density_valid = np.array([hist[xi, yi] for xi, yi in zip(x_idx, y_idx)], dtype=float)
    density[valid] = density_valid
    return np.argsort(density)


#-------------------------데이터 EDA, 구조 파악-------------------------------------

# 데이터프레임의 기본 구조를 확인하는 함수
def summarize_dataframe(df: pd.DataFrame, head_n: int = 5) -> None:
    with pd.option_context(
        'display.max_columns', None,
        'display.width', 2000,
        'display.max_colwidth', None,
        'display.expand_frame_repr', False,
        'display.max_rows', None
    ):
        print("[Shape]")
        print("-" * 50)
        print(f"rows: {df.shape[0]}, cols: {df.shape[1]}")
        print()

        print("[Columns]")
        print("-" * 50)
        print(df.columns.tolist())
        print()

        print("[Head]")
        print("-" * 50)
        print(df.head(head_n))
        print()

        print("[Info]")
        print("-" * 50)
        df.info()
        print()

        print("[Describe]")
        print("-" * 50)
        print(df.describe(include="all").T)
        print()

#------------------------------[결측 처리]------------------------------------

# 결측치 개수와 비율을 표 형태로 반환하는 함수
def na_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_ratio": df.isna().mean()
    })
    print("[NA_Summary]")
    print("-" * 50)
    return result.sort_values("missing_ratio", ascending=False), print(), print(f'결측치 포함 행 개수: {df.isna().any(axis=1).sum()}'), print(f'결측치 포함 행 비율: {df.isna().any(axis=1).mean()}'), print()


# 결측치 칼럼 간 공결측 패턴 분석 함수
    # 사용방법: 함수 안에 원하는 df 파일 -> 테이블 자동 생성
def na_copattern(df: pd.DataFrame) -> None:
    na_columns = df.columns[df.isna().any()]

    print('[NA co-pattern summary]')
    print('-' * 50)

    if len(na_columns) == 0:
        print('결측치가 있는 컬럼이 없습니다.')
        print()
        return

    # 결측 플래그 생성
    na_pattern = df[na_columns].isna()

    # 결측이 1개 이상 있는 행만 남김
    na_pattern_only = na_pattern[na_pattern.any(axis=1)]

    if na_pattern_only.empty:
        print('결측 발생 행이 없습니다.')
        print()
        return

    # 결측 패턴별 count
    counts = na_pattern_only.value_counts(ascending=False)

    # 결측 발생 행 대비 비율
    rates = na_pattern_only.value_counts(ascending=False, normalize=True) * 100

    result = pd.concat([counts, rates], axis=1)
    result.columns = ['NA_count', 'NA_rate(%)']
    result = result.reset_index()
    result = result.replace({True: np.nan, False: 'X'})

    with pd.option_context(
        'display.max_columns', None,
        'display.width', 2000,
        'display.max_colwidth', None
    ):
        print(result)

    print()
    
    return result


# 결측치 프로파일링(수치형+범주형 포함)
    # 사용 방법: na_profile(데이터프레임)
def na_profile(
    df: pd.DataFrame,
    exclude_cols: list[str] = None,
    d_threshold: float = 0.7,
    abs_gap_threshold: float = 10.0,
    rel_gap_threshold: float = 1.5,
    min_count_threshold: int = 20
) -> None:
    if exclude_cols is None:
        exclude_cols = []

    na_columns = [col for col in df.columns if df[col].isna().any()]
    if len(na_columns) == 0:
        print('[NA profile]')
        print('-' * 80)
        print('결측집단이 있는 컬럼이 없습니다.')
        print()
        return

    tmp = df.copy()
    for col in na_columns:
        flag_col = f'na_{col}'
        if flag_col not in tmp.columns:
            tmp[flag_col] = tmp[col].isna()

    na_flags = [f'na_{col}' for col in na_columns]

    num_cols = tmp.select_dtypes(include='number').columns.tolist()
    cat_cols = tmp.select_dtypes(exclude='number').columns.tolist()

    num_cols = [col for col in num_cols if not str(col).startswith('na_') and col not in exclude_cols]
    cat_cols = [col for col in cat_cols if not str(col).startswith('na_')]

    print('[NA profile - summary]')
    print('=' * 80)

    for na in na_flags:
        target_col = na.replace('na_', '')
        use_num_cols = [col for col in num_cols if col != target_col]
        use_cat_cols = [col for col in cat_cols if col != target_col]

        num_summary_rows = []
        for col in use_num_cols:
            sub = tmp[[col, na]].dropna().copy()
            if sub[na].nunique() != 2:
                continue

            grp = sub.groupby(na)[col]
            mean_non_missing = grp.mean().get(False, np.nan)
            mean_missing = grp.mean().get(True, np.nan)
            std_non_missing = grp.std().get(False, np.nan)
            std_missing = grp.std().get(True, np.nan)
            n_non_missing = grp.count().get(False, 0)
            n_missing = grp.count().get(True, 0)

            pooled_std = np.sqrt(
                (((n_non_missing - 1) * (std_non_missing ** 2)) + ((n_missing - 1) * (std_missing ** 2)))
                / (n_non_missing + n_missing - 2)
            ) if (n_non_missing > 1 and n_missing > 1 and (n_non_missing + n_missing - 2) > 0) else np.nan

            cohens_d = (
                (mean_missing - mean_non_missing) / pooled_std
                if pd.notna(pooled_std) and pooled_std != 0 else np.nan
            )
            abs_d = abs(cohens_d) if pd.notna(cohens_d) else np.nan

            if pd.notna(abs_d) and abs_d >= d_threshold:
                num_summary_rows.append({
                    'variable': col,
                    '결측집단_mean': round(mean_missing, 3),
                    '비결측집단_mean': round(mean_non_missing, 3),
                    'mean_diff': round(mean_missing - mean_non_missing, 3),
                    'cohens_d': round(cohens_d, 3),
                    'abs_cohens_d': round(abs_d, 3)
                })

        num_summary_df = pd.DataFrame(num_summary_rows)
        if len(num_summary_df) > 0:
            num_summary_df = num_summary_df.sort_values('abs_cohens_d', ascending=False)

        print(f'[{na} - summary - numeric]')
        print('-' * 80)
        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 2000,
            'display.expand_frame_repr', False
        ):
            if len(num_summary_df) > 0:
                print(num_summary_df.to_string(index=False))
            else:
                print('기준 충족 변수 없음')
        print()

        cat_summary_rows = []
        tmp_flag = tmp.copy()
        tmp_flag[na] = tmp_flag[na].map({False: '비결측집단', True: '결측집단'})

        for col in use_cat_cols:
            sub = tmp_flag[[col, na]].copy()
            count_table = pd.crosstab(sub[col], sub[na], dropna=False)
            rate_table = pd.crosstab(sub[col], sub[na], normalize='columns', dropna=False).mul(100)

            if {'결측집단', '비결측집단'}.issubset(rate_table.columns):
                total_count = count_table.sum(axis=1)
                gap = (rate_table['결측집단'] - rate_table['비결측집단']).abs()

                candidates = pd.DataFrame({
                    'count': total_count,
                    '결측집단_%': rate_table['결측집단'],
                    '비결측집단_%': rate_table['비결측집단'],
                    'abs_gap_%p': gap
                })

                candidates = candidates[candidates['count'] >= min_count_threshold]
                if len(candidates) == 0:
                    continue

                top_category = candidates['abs_gap_%p'].idxmax()
                missing_rate = candidates.loc[top_category, '결측집단_%']
                non_missing_rate = candidates.loc[top_category, '비결측집단_%']

                smaller = min(missing_rate, non_missing_rate)
                bigger = max(missing_rate, non_missing_rate)
                rel_gap = (bigger / smaller) if smaller > 0 else np.nan

                if (
                    pd.notna(rel_gap)
                    and candidates.loc[top_category, 'abs_gap_%p'] >= abs_gap_threshold
                    and rel_gap >= rel_gap_threshold
                ):
                    cat_summary_rows.append({
                        'variable': col,
                        'top_category': top_category,
                        'count': int(candidates.loc[top_category, 'count']),
                        '결측집단_%': round(missing_rate, 2),
                        '비결측집단_%': round(non_missing_rate, 2),
                        'abs_gap_%p': round(candidates.loc[top_category, 'abs_gap_%p'], 2),
                        'rel_gap_x': round(rel_gap, 3)
                    })

        cat_summary_df = pd.DataFrame(cat_summary_rows)
        if len(cat_summary_df) > 0:
            cat_summary_df = cat_summary_df.sort_values(['abs_gap_%p', 'rel_gap_x'], ascending=[False, False])

        print(f'[{na} - summary - categorical]')
        print('-' * 80)
        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 2000,
            'display.expand_frame_repr', False
        ):
            if len(cat_summary_df) > 0:
                print(cat_summary_df.to_string(index=False))
            else:
                print('기준 충족 변수 없음')
        print()

    for na in na_flags:
        target_col = na.replace('na_', '')
        use_num_cols = [col for col in num_cols if col != target_col]
        use_cat_cols = [col for col in cat_cols if col != target_col]

        print(f'[{na}]')
        print('=' * 80)

        tmp_flag = tmp.copy()
        tmp_flag[na] = tmp_flag[na].map({False: '비결측집단', True: '결측집단'})

        if use_num_cols:
            print('[numeric profiling]')
            print('-' * 80)

            agg_result = tmp_flag.groupby(na)[use_num_cols].agg(['mean', 'median', 'std']).round(2)
            agg_result = agg_result.reindex(['결측집단', '비결측집단'])

            for stat in ['mean', 'median', 'std']:
                print(f'[{na} - {stat}]')
                stat_table = agg_result.xs(stat, axis=1, level=1).T.reset_index()
                stat_table.columns = ['variable', '결측집단', '비결측집단']

                with pd.option_context(
                    'display.max_rows', None,
                    'display.max_columns', None,
                    'display.width', 2000,
                    'display.expand_frame_repr', False
                ):
                    print(stat_table.to_string(index=False))
                print()

        if use_cat_cols:
            print('[categorical profiling]')
            print('-' * 80)

            for col in use_cat_cols:
                print(f'[{na} vs {col}]')

                count_table = pd.crosstab(tmp_flag[col], tmp_flag[na], dropna=False)
                rate_table = pd.crosstab(
                    tmp_flag[col], tmp_flag[na], normalize='columns', dropna=False
                ).mul(100).round(2)

                count_table = count_table.reindex(columns=['결측집단', '비결측집단'], fill_value=0)
                rate_table = rate_table.reindex(columns=['결측집단', '비결측집단'], fill_value=0)

                rate_table_fmt = rate_table.map(lambda x: f'{x:.2f}%')
                merged_table = count_table.astype(str) + '(' + rate_table_fmt + ')'

                with pd.option_context(
                    'display.max_rows', None,
                    'display.max_columns', None,
                    'display.width', 2000,
                    'display.expand_frame_repr', False
                ):
                    print(merged_table)
                print()

        print('=' * 80)
        print()
        

# 비결측 변수와 결측 집단 연관성 분석 함수(수치형+범주형 포함)
    # 사용 방법: 원본 df, 수치형(pd.nunique()>int로 범주화 가능한) 칼럼, 범주형 칼럼, qcut)
def na_association_profile(
    df: pd.DataFrame,
    num_cols: list,
    cat_cols: list,
    q: int = 5,
    d_threshold: float = 0.7,
    abs_gap_threshold: float = 10.0,
    rel_gap_threshold: float = 1.5,
    min_count_threshold: int = 20
) -> None:
    tmp = df.copy()

    base_na_columns = [col for col in tmp.columns if tmp[col].isna().any() and not str(col).startswith('na_')]
    for col in base_na_columns:
        flag_col = f'na_{col}'
        if flag_col not in tmp.columns:
            tmp[flag_col] = tmp[col].isna()

    na_flag_cols = [f'na_{col}' for col in base_na_columns if f'na_{col}' in tmp.columns]

    if len(na_flag_cols) == 0:
        print('[NA association profile]')
        print('-' * 80)
        print('결측집단이 있는 컬럼이 없습니다.')
        print()
        return

    print('[NA association profile - summary]')
    print('=' * 80)

    for na in na_flag_cols:
        target_col = na.replace('na_', '')
        use_num_cols = [col for col in num_cols if col != target_col and col in tmp.columns and not str(col).startswith('na_')]
        use_cat_cols = [col for col in cat_cols if col != target_col and col in tmp.columns and not str(col).startswith('na_')]

        num_summary_rows = []
        for col in use_num_cols:
            sub = tmp[[col, na]].dropna().copy()
            if sub[na].nunique() != 2:
                continue

            if sub[col].nunique() <= 10:
                group_col = col
            else:
                try:
                    sub[f'{col}_bin'] = pd.qcut(sub[col], q=q, duplicates='drop')
                    group_col = f'{col}_bin'
                except ValueError:
                    continue

            count_table = pd.crosstab(sub[group_col], sub[na], dropna=False)
            rate_table = pd.crosstab(sub[group_col], sub[na], normalize='index', dropna=False).mul(100)

            if True not in rate_table.columns:
                continue

            total_n = len(sub)
            total_missing = (sub[na] == True).sum()

            candidates = []
            for grp in rate_table.index:
                grp_n = int(count_table.loc[grp].sum())
                if grp_n < min_count_threshold:
                    continue

                grp_missing = int(count_table.loc[grp, True]) if True in count_table.columns else 0
                other_n = total_n - grp_n
                other_missing = total_missing - grp_missing
                if other_n <= 0:
                    continue

                grp_missing_rate = (grp_missing / grp_n) * 100
                other_missing_rate = (other_missing / other_n) * 100
                abs_gap = abs(grp_missing_rate - other_missing_rate)

                smaller = min(grp_missing_rate, other_missing_rate)
                bigger = max(grp_missing_rate, other_missing_rate)
                rel_gap = (bigger / smaller) if smaller > 0 else np.nan

                if (
                    pd.notna(rel_gap)
                    and abs_gap >= abs_gap_threshold
                    and rel_gap >= rel_gap_threshold
                ):
                    candidates.append({
                        'variable': col,
                        'top_group': grp,
                        'count': grp_n,
                        '결측집단_%': round(grp_missing_rate, 2),
                        '비결측집단_%': round(100 - grp_missing_rate, 2),
                        'other_결측집단_%': round(other_missing_rate, 2),
                        'other_비결측집단_%': round(100 - other_missing_rate, 2),
                        'abs_gap_%p': round(abs_gap, 2),
                        'rel_gap_x': round(rel_gap, 3)
                    })

            if len(candidates) > 0:
                best_row = (
                    pd.DataFrame(candidates)
                    .sort_values(['abs_gap_%p', 'rel_gap_x'], ascending=[False, False])
                    .iloc[0]
                    .to_dict()
                )
                num_summary_rows.append(best_row)

        num_summary_df = pd.DataFrame(num_summary_rows)
        if len(num_summary_df) > 0:
            num_summary_df = num_summary_df.sort_values(['abs_gap_%p', 'rel_gap_x'], ascending=[False, False])

        print(f'[{na} - summary - numeric]')
        print('-' * 80)
        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 2000,
            'display.expand_frame_repr', False
        ):
            if len(num_summary_df) > 0:
                print(num_summary_df.to_string(index=False))
            else:
                print('기준 충족 변수 없음')
        print()

        cat_summary_rows = []
        for col in use_cat_cols:
            sub = tmp[[col, na]].copy()

            count_table = pd.crosstab(sub[col], sub[na], dropna=False)
            rate_table = pd.crosstab(sub[col], sub[na], normalize='index', dropna=False).mul(100)

            if True not in rate_table.columns:
                continue

            total_n = len(sub)
            total_missing = (sub[na] == True).sum()

            candidates = []
            for grp in rate_table.index:
                grp_n = int(count_table.loc[grp].sum())
                if grp_n < min_count_threshold:
                    continue

                grp_missing = int(count_table.loc[grp, True]) if True in count_table.columns else 0
                other_n = total_n - grp_n
                other_missing = total_missing - grp_missing
                if other_n <= 0:
                    continue

                grp_missing_rate = (grp_missing / grp_n) * 100
                other_missing_rate = (other_missing / other_n) * 100
                abs_gap = abs(grp_missing_rate - other_missing_rate)

                smaller = min(grp_missing_rate, other_missing_rate)
                bigger = max(grp_missing_rate, other_missing_rate)
                rel_gap = (bigger / smaller) if smaller > 0 else np.nan

                if (
                    pd.notna(rel_gap)
                    and abs_gap >= abs_gap_threshold
                    and rel_gap >= rel_gap_threshold
                ):
                    candidates.append({
                        'variable': col,
                        'top_category': grp,
                        'count': grp_n,
                        '결측집단_%': round(grp_missing_rate, 2),
                        '비결측집단_%': round(100 - grp_missing_rate, 2),
                        'other_결측집단_%': round(other_missing_rate, 2),
                        'other_비결측집단_%': round(100 - other_missing_rate, 2),
                        'abs_gap_%p': round(abs_gap, 2),
                        'rel_gap_x': round(rel_gap, 3)
                    })

            if len(candidates) > 0:
                best_row = (
                    pd.DataFrame(candidates)
                    .sort_values(['abs_gap_%p', 'rel_gap_x'], ascending=[False, False])
                    .iloc[0]
                    .to_dict()
                )
                cat_summary_rows.append(best_row)

        cat_summary_df = pd.DataFrame(cat_summary_rows)
        if len(cat_summary_df) > 0:
            cat_summary_df = cat_summary_df.sort_values(['abs_gap_%p', 'rel_gap_x'], ascending=[False, False])

        print(f'[{na} - summary - categorical]')
        print('-' * 80)
        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 2000,
            'display.expand_frame_repr', False
        ):
            if len(cat_summary_df) > 0:
                print(cat_summary_df.to_string(index=False))
            else:
                print('기준 충족 변수 없음')
        print()

    for na in na_flag_cols:
        target_col = na.replace('na_', '')
        use_num_cols = [col for col in num_cols if col != target_col and col in tmp.columns and not str(col).startswith('na_')]
        use_cat_cols = [col for col in cat_cols if col != target_col and col in tmp.columns and not str(col).startswith('na_')]

        print(f'[{na}]')
        print('=' * 80)

        if use_num_cols:
            print('[수치형 변수 -> 결측집단 변수]')
            print('-' * 80)

            for col in use_num_cols:
                print(f'[{na} vs {col}]')
                sub = tmp[[col, na]].dropna().copy()

                if sub[col].nunique() <= 10:
                    count_table = pd.crosstab(sub[col], sub[na], dropna=False)
                    rate_table = pd.crosstab(sub[col], sub[na], normalize='index', dropna=False).mul(100).round(1)
                else:
                    sub[f'{col}_bin'] = pd.qcut(sub[col], q=q, duplicates='drop')
                    count_table = pd.crosstab(sub[f'{col}_bin'], sub[na], dropna=False)
                    rate_table = pd.crosstab(sub[f'{col}_bin'], sub[na], normalize='index', dropna=False).mul(100).round(1)

                count_table = count_table.rename(columns={True: '결측집단', False: '비결측집단'})
                rate_table = rate_table.rename(columns={True: '결측집단', False: '비결측집단'})

                count_table = count_table.reindex(columns=['결측집단', '비결측집단'], fill_value=0)
                rate_table = rate_table.reindex(columns=['결측집단', '비결측집단'], fill_value=0)

                rate_table_fmt = rate_table.map(lambda x: f'{x:.1f}%')
                merged_table = count_table.astype(str) + '(' + rate_table_fmt + ')'

                with pd.option_context(
                    'display.max_rows', None,
                    'display.max_columns', None,
                    'display.width', 2000,
                    'display.expand_frame_repr', False
                ):
                    print(merged_table)
                print()

        if use_cat_cols:
            print('[범주형 변수 -> 결측집단 변수]')
            print('-' * 80)

            for col in use_cat_cols:
                print(f'[{na} vs {col}]')
                sub = tmp[[col, na]].copy()

                count_table = pd.crosstab(sub[col], sub[na], dropna=False)
                rate_table = pd.crosstab(sub[col], sub[na], normalize='index', dropna=False).mul(100).round(1)

                count_table = count_table.rename(columns={True: '결측집단', False: '비결측집단'})
                rate_table = rate_table.rename(columns={True: '결측집단', False: '비결측집단'})

                count_table = count_table.reindex(columns=['결측집단', '비결측집단'], fill_value=0)
                rate_table = rate_table.reindex(columns=['결측집단', '비결측집단'], fill_value=0)

                rate_table_fmt = rate_table.map(lambda x: f'{x:.1f}%')
                merged_table = count_table.astype(str) + '(' + rate_table_fmt + ')'

                with pd.option_context(
                    'display.max_rows', None,
                    'display.max_columns', None,
                    'display.width', 2000,
                    'display.expand_frame_repr', False
                ):
                    print(merged_table)
                print()

        print('=' * 80)
        print()


#------------------------------[중복 처리]------------------------------------

# 중복 행 개수와 비율을 요약해서 반환하는 함수
def duplicate_summary(df: pd.DataFrame) -> pd.DataFrame:
    dup_count = int(df.duplicated().sum())
    dup_ratio = dup_count / len(df) if len(df) > 0 else np.nan
    
    print('[duplicate_summary]')
    print('-'*50)
    return pd.DataFrame({
        "duplicate_count": [dup_count],
        "duplicate_ratio": [dup_ratio]
    })


#------------------------------[이상치 처리]------------------------------------

# 하나의 수치형 시리즈에 대해 IQR 기준 이상치 경계와 이상치만 반환하는 함수
def iqr_outlier_bound_mask(series: pd.Series):
    series_clean = series.dropna()

    q1 = series_clean.quantile(0.25)
    q3 = series_clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (series_clean < lower) | (series_clean > upper)
    outlier_df = pd.DataFrame(series_clean[mask].sort_values(ascending=False))

    return lower, upper, mask, outlier_df


# 데이터프레임의 수치형 컬럼별 이상치 개수와 비율을 요약표로 반환하는 함수
def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=np.number).columns
    rows = []

    for col in numeric_cols:
        series = df[col].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)

        rows.append({
            "column": col,
            "lower": lower,
            "upper": upper,
            "outlier_count": int(mask.sum()),
            "outlier_ratio(%)": float(mask.mean())*100 if len(series) > 0 else np.nan
        })

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values("outlier_count", ascending=False)
    
    print('[outlier_summary]')
    print('-'*80)
    print(result)


# 수치형 컬럼의 히스토그램과 박스플롯을 컬럼별로 시각화하는 함수
def plot_numeric_distributions(df: pd.DataFrame, cols=None, bins: int = 30) -> None:
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns

    for col in cols:
        series = df[col].dropna()

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.hist(series, bins=bins)
        plt.title(f"{col} histogram")

        plt.subplot(1, 2, 2)
        plt.boxplot(series, vert=False)
        plt.title(f"{col} boxplot")

        plt.tight_layout()
        plt.show()


# 여러 개의 수치형 컬럼 중 원하는 컬럼만 이상치 상세값으로 확인하는 함수
def get_outlier_values(df: pd.DataFrame, col: str) -> pd.DataFrame:
    lower, upper, mask, outlier_df = iqr_outlier_bound_mask(df[col])
    return outlier_df


#------------------------------[타깃 EDA]------------------------------------

# 타깃 변수의 분포와 비율을 확인하는 함수
def check_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    target_count = df[target_col].value_counts(dropna=False).sort_index()
    target_ratio = df[target_col].value_counts(dropna=False, normalize=True).sort_index()

    target_table = pd.DataFrame({
        'count': target_count,
        'ratio': target_ratio.round(4)*100
    })

    print('[Target Check]')
    print('=' * 50)
    print(f'target column: {target_col}')
    print('-' * 50)
    print(target_table)
    print('-' * 50)
    print(f'total rows: {len(df)}')
    print(f'target ratio: {df[target_col].mean()*100:.4f}')
    print()

    if len(target_count) == 2:
        majority_ratio = target_ratio.max()
        minority_ratio = target_ratio.min()
        print('[Target Imbalance Check]')
        print('=' * 50)
        print(f'majority class ratio: {majority_ratio*100:.4f}')
        print(f'minority class ratio: {minority_ratio*100:.4f}')
        print(f'baseline accuracy: {majority_ratio*100:.4f}')
        print()

    return target_table

# 타깃 프로파일링 및 변수-타깃 관계 리턴 함수
    #사용방법: orginaDF + TARGET + Q(int for q cut)
def target_profile(
    df: pd.DataFrame,
    target_col: str,
    q: int = 5,
    d_threshold: float = 0.7,
    abs_gap_threshold: float = 10.0,
    rel_gap_threshold: float = 1.5,
    min_count_threshold: int = 20
) -> None:
    tmp = df.copy()

    if target_col not in tmp.columns:
        print('[Target profile]')
        print('-' * 80)
        print(f'타깃 컬럼 {target_col} 이(가) 데이터프레임에 없습니다.')
        print()
        return

    tmp = tmp.dropna(subset=[target_col]).copy()

    num_cols = tmp.select_dtypes(include='number').columns.tolist()
    cat_cols = tmp.select_dtypes(exclude='number').columns.tolist()

    num_cols = [col for col in num_cols if col != target_col]
    cat_cols = [col for col in cat_cols if col != target_col]

    target_values = sorted(tmp[target_col].dropna().unique().tolist())

    print('[Target profile]')
    print('=' * 80)
    print(f'target: {target_col}')
    print('=' * 80)

    # ------------------------------------------------------------------
    # 1) 프로파일링 summary
    # ------------------------------------------------------------------
    print('[profiling summary]')
    print('=' * 80)

    if len(target_values) == 2:
        t0, t1 = target_values[0], target_values[1]

        num_summary_rows = []
        for col in num_cols:
            sub = tmp[[col, target_col]].dropna().copy()

            grp = sub.groupby(target_col)[col]
            mean_0 = grp.mean().get(t0, np.nan)
            mean_1 = grp.mean().get(t1, np.nan)
            std_0 = grp.std().get(t0, np.nan)
            std_1 = grp.std().get(t1, np.nan)
            n_0 = grp.count().get(t0, 0)
            n_1 = grp.count().get(t1, 0)

            pooled_std = np.sqrt(
                (((n_0 - 1) * (std_0 ** 2)) + ((n_1 - 1) * (std_1 ** 2))) / (n_0 + n_1 - 2)
            ) if (n_0 > 1 and n_1 > 1 and (n_0 + n_1 - 2) > 0) else np.nan

            cohens_d = (mean_1 - mean_0) / pooled_std if pd.notna(pooled_std) and pooled_std != 0 else np.nan
            abs_d = abs(cohens_d) if pd.notna(cohens_d) else np.nan

            if pd.notna(abs_d) and abs_d >= d_threshold:
                num_summary_rows.append({
                    'variable': col,
                    f'{target_col}={t0}_mean': round(mean_0, 3),
                    f'{target_col}={t1}_mean': round(mean_1, 3),
                    'mean_diff': round(mean_1 - mean_0, 3),
                    'cohens_d': round(cohens_d, 3),
                    'abs_cohens_d': round(abs_d, 3)
                })

        num_summary_df = pd.DataFrame(num_summary_rows)
        if len(num_summary_df) > 0:
            num_summary_df = num_summary_df.sort_values('abs_cohens_d', ascending=False)

        print('[profiling summary - numeric]')
        print('-' * 80)
        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 2000,
            'display.expand_frame_repr', False
        ):
            if len(num_summary_df) > 0:
                print(num_summary_df.to_string(index=False))
            else:
                print('기준 충족 변수 없음')
        print()

        cat_summary_rows = []
        for col in cat_cols:
            sub = tmp[[col, target_col]].copy()

            count_table = pd.crosstab(sub[col], sub[target_col], dropna=False)
            rate_table = pd.crosstab(sub[col], sub[target_col], normalize='columns', dropna=False).mul(100)

            if {t0, t1}.issubset(rate_table.columns):
                total_count = count_table.sum(axis=1)
                gap = (rate_table[t1] - rate_table[t0]).abs()

                candidates = pd.DataFrame({
                    'count': total_count,
                    f'{target_col}={t0}_%': rate_table[t0],
                    f'{target_col}={t1}_%': rate_table[t1],
                    'abs_gap_%p': gap
                })

                candidates = candidates[candidates['count'] >= min_count_threshold]
                if len(candidates) == 0:
                    continue

                top_category = candidates['abs_gap_%p'].idxmax()
                rate_0 = candidates.loc[top_category, f'{target_col}={t0}_%']
                rate_1 = candidates.loc[top_category, f'{target_col}={t1}_%']

                smaller = min(rate_0, rate_1)
                bigger = max(rate_0, rate_1)
                rel_gap = (bigger / smaller) if smaller > 0 else np.nan

                if (
                    pd.notna(rel_gap)
                    and candidates.loc[top_category, 'abs_gap_%p'] >= abs_gap_threshold
                    and rel_gap >= rel_gap_threshold
                ):
                    cat_summary_rows.append({
                        'variable': col,
                        'top_category': top_category,
                        'count': int(candidates.loc[top_category, 'count']),
                        f'{target_col}={t0}_%': round(rate_0, 2),
                        f'{target_col}={t1}_%': round(rate_1, 2),
                        'abs_gap_%p': round(candidates.loc[top_category, 'abs_gap_%p'], 2),
                        'rel_gap_x': round(rel_gap, 3)
                    })

        cat_summary_df = pd.DataFrame(cat_summary_rows)
        if len(cat_summary_df) > 0:
            cat_summary_df = cat_summary_df.sort_values(['abs_gap_%p', 'rel_gap_x'], ascending=[False, False])

        print('[profiling summary - categorical]')
        print('-' * 80)
        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 2000,
            'display.expand_frame_repr', False
        ):
            if len(cat_summary_df) > 0:
                print(cat_summary_df.to_string(index=False))
            else:
                print('기준 충족 변수 없음')
        print()
    else:
        print('[profiling summary - numeric]')
        print('-' * 80)
        print('binary target에서만 생성')
        print()

        print('[profiling summary - categorical]')
        print('-' * 80)
        print('binary target에서만 생성')
        print()

    # ------------------------------------------------------------------
    # 2) 프로파일링 상세
    # ------------------------------------------------------------------
    if num_cols:
        print('[target profiling - numeric]')
        print('-' * 80)

        agg_result = tmp.groupby(target_col)[num_cols].agg(['mean', 'median', 'std']).round(2)
        agg_result = agg_result.reindex(target_values)

        for stat in ['mean', 'median', 'std']:
            print(f'[{target_col} - {stat}]')
            stat_table = agg_result.xs(stat, axis=1, level=1).T.reset_index()
            stat_table.columns = ['variable'] + [f'{target_col}={v}' for v in target_values]

            with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
                'display.width', 2000,
                'display.expand_frame_repr', False
            ):
                print(stat_table.to_string(index=False))
            print()

    if cat_cols:
        print('[target profiling - categorical]')
        print('-' * 80)

        for col in cat_cols:
            print(f'[{target_col} vs {col}]')
            count_table = pd.crosstab(tmp[col], tmp[target_col], dropna=False)
            rate_table = pd.crosstab(tmp[col], tmp[target_col], normalize='columns', dropna=False).mul(100).round(1)

            count_table = count_table.reindex(columns=target_values, fill_value=0)
            rate_table = rate_table.reindex(columns=target_values, fill_value=0)

            count_table.columns = [f'{target_col}={c}' for c in target_values]
            rate_table.columns = [f'{target_col}={c}' for c in target_values]

            rate_table_fmt = rate_table.map(lambda x: f'{x:.1f}%')
            merged_table = count_table.astype(str) + '(' + rate_table_fmt + ')'

            with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
                'display.width', 2000,
                'display.expand_frame_repr', False
            ):
                print(merged_table)
            print()

    # ------------------------------------------------------------------
    # 3) 관계분석 summary
    # ------------------------------------------------------------------
    print('[association summary]')
    print('=' * 80)

    if len(target_values) == 2:
        t0, t1 = target_values[0], target_values[1]

        assoc_num_rows = []
        for col in num_cols:
            sub = tmp[[col, target_col]].dropna().copy()

            if sub[col].nunique() <= 10:
                group_col = col
            else:
                try:
                    sub[f'{col}_bin'] = pd.qcut(sub[col], q=q, duplicates='drop')
                    group_col = f'{col}_bin'
                except ValueError:
                    continue

            count_table = pd.crosstab(sub[group_col], sub[target_col], dropna=False)
            rate_table = pd.crosstab(sub[group_col], sub[target_col], normalize='index', dropna=False).mul(100)

            if t1 not in rate_table.columns:
                continue

            total_n = len(sub)
            total_pos = (sub[target_col] == t1).sum()

            candidates = []
            for grp in rate_table.index:
                grp_n = int(count_table.loc[grp].sum())
                if grp_n < min_count_threshold:
                    continue

                grp_pos = int(count_table.loc[grp, t1]) if t1 in count_table.columns else 0
                other_n = total_n - grp_n
                other_pos = total_pos - grp_pos
                if other_n <= 0:
                    continue

                grp_rate = (grp_pos / grp_n) * 100
                other_rate = (other_pos / other_n) * 100
                abs_gap = abs(grp_rate - other_rate)

                smaller = min(grp_rate, other_rate)
                bigger = max(grp_rate, other_rate)
                rel_gap = (bigger / smaller) if smaller > 0 else np.nan

                if (
                    pd.notna(rel_gap)
                    and abs_gap >= abs_gap_threshold
                    and rel_gap >= rel_gap_threshold
                ):
                    candidates.append({
                        'variable': col,
                        'top_group': grp,
                        'count': grp_n,
                        f'{target_col}={t1}_%': round(grp_rate, 2),
                        f'{target_col}!={t1}_%': round(100 - grp_rate, 2),
                        f'other_{target_col}={t1}_%': round(other_rate, 2),
                        f'other_{target_col}!={t1}_%': round(100 - other_rate, 2),
                        'abs_gap_%p': round(abs_gap, 2),
                        'rel_gap_x': round(rel_gap, 3)
                    })

            if len(candidates) > 0:
                best_row = (
                    pd.DataFrame(candidates)
                    .sort_values(['abs_gap_%p', 'rel_gap_x'], ascending=[False, False])
                    .iloc[0]
                    .to_dict()
                )
                assoc_num_rows.append(best_row)

        assoc_num_df = pd.DataFrame(assoc_num_rows)
        if len(assoc_num_df) > 0:
            assoc_num_df = assoc_num_df.sort_values(['abs_gap_%p', 'rel_gap_x'], ascending=[False, False])

        print('[association summary - numeric]')
        print('-' * 80)
        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 2000,
            'display.expand_frame_repr', False
        ):
            if len(assoc_num_df) > 0:
                print(assoc_num_df.to_string(index=False))
            else:
                print('기준 충족 변수 없음')
        print()

        assoc_cat_rows = []
        for col in cat_cols:
            sub = tmp[[col, target_col]].copy()

            count_table = pd.crosstab(sub[col], sub[target_col], dropna=False)
            rate_table = pd.crosstab(sub[col], sub[target_col], normalize='index', dropna=False).mul(100)

            if t1 not in rate_table.columns:
                continue

            total_n = len(sub)
            total_pos = (sub[target_col] == t1).sum()

            candidates = []
            for grp in rate_table.index:
                grp_n = int(count_table.loc[grp].sum())
                if grp_n < min_count_threshold:
                    continue

                grp_pos = int(count_table.loc[grp, t1]) if t1 in count_table.columns else 0
                other_n = total_n - grp_n
                other_pos = total_pos - grp_pos
                if other_n <= 0:
                    continue

                grp_rate = (grp_pos / grp_n) * 100
                other_rate = (other_pos / other_n) * 100
                abs_gap = abs(grp_rate - other_rate)

                smaller = min(grp_rate, other_rate)
                bigger = max(grp_rate, other_rate)
                rel_gap = (bigger / smaller) if smaller > 0 else np.nan

                if (
                    pd.notna(rel_gap)
                    and abs_gap >= abs_gap_threshold
                    and rel_gap >= rel_gap_threshold
                ):
                    candidates.append({
                        'variable': col,
                        'top_category': grp,
                        'count': grp_n,
                        f'{target_col}={t1}_%': round(grp_rate, 2),
                        f'{target_col}!={t1}_%': round(100 - grp_rate, 2),
                        f'other_{target_col}={t1}_%': round(other_rate, 2),
                        f'other_{target_col}!={t1}_%': round(100 - other_rate, 2),
                        'abs_gap_%p': round(abs_gap, 2),
                        'rel_gap_x': round(rel_gap, 3)
                    })

            if len(candidates) > 0:
                best_row = (
                    pd.DataFrame(candidates)
                    .sort_values(['abs_gap_%p', 'rel_gap_x'], ascending=[False, False])
                    .iloc[0]
                    .to_dict()
                )
                assoc_cat_rows.append(best_row)

        assoc_cat_df = pd.DataFrame(assoc_cat_rows)
        if len(assoc_cat_df) > 0:
            assoc_cat_df = assoc_cat_df.sort_values(['abs_gap_%p', 'rel_gap_x'], ascending=[False, False])

        print('[association summary - categorical]')
        print('-' * 80)
        with pd.option_context(
            'display.max_rows', None,
            'display.max_columns', None,
            'display.width', 2000,
            'display.expand_frame_repr', False
        ):
            if len(assoc_cat_df) > 0:
                print(assoc_cat_df.to_string(index=False))
            else:
                print('기준 충족 변수 없음')
        print()
    else:
        print('[association summary - numeric]')
        print('-' * 80)
        print('binary target에서만 생성')
        print()

        print('[association summary - categorical]')
        print('-' * 80)
        print('binary target에서만 생성')
        print()

    # ------------------------------------------------------------------
    # 4) 관계분석 상세
    # ------------------------------------------------------------------
    if num_cols:
        print('[numeric variable -> target]')
        print('-' * 80)

        for col in num_cols:
            print(f'[{col} -> {target_col}]')
            sub = tmp[[col, target_col]].dropna().copy()

            if sub[col].nunique() <= 10:
                count_table = pd.crosstab(sub[col], sub[target_col], dropna=False)
                rate_table = pd.crosstab(sub[col], sub[target_col], normalize='index', dropna=False).mul(100).round(1)
            else:
                sub[f'{col}_bin'] = pd.qcut(sub[col], q=q, duplicates='drop')
                count_table = pd.crosstab(sub[f'{col}_bin'], sub[target_col], dropna=False)
                rate_table = pd.crosstab(sub[f'{col}_bin'], sub[target_col], normalize='index', dropna=False).mul(100).round(1)

            count_table = count_table.reindex(columns=target_values, fill_value=0)
            rate_table = rate_table.reindex(columns=target_values, fill_value=0)

            count_table.columns = [f'{target_col}={c}' for c in target_values]
            rate_table.columns = [f'{target_col}={c}' for c in target_values]

            rate_table_fmt = rate_table.map(lambda x: f'{x:.1f}%')
            merged_table = count_table.astype(str) + '(' + rate_table_fmt + ')'

            with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
                'display.width', 2000,
                'display.expand_frame_repr', False
            ):
                print(merged_table)
            print()

    if cat_cols:
        print('[categorical variable -> target]')
        print('-' * 80)

        for col in cat_cols:
            print(f'[{col} -> {target_col}]')
            sub = tmp[[col, target_col]].copy()

            count_table = pd.crosstab(sub[col], sub[target_col], dropna=False)
            rate_table = pd.crosstab(sub[col], sub[target_col], normalize='index', dropna=False).mul(100).round(1)

            count_table = count_table.reindex(columns=target_values, fill_value=0)
            rate_table = rate_table.reindex(columns=target_values, fill_value=0)

            count_table.columns = [f'{target_col}={c}' for c in target_values]
            rate_table.columns = [f'{target_col}={c}' for c in target_values]

            rate_table_fmt = rate_table.map(lambda x: f'{x:.1f}%')
            merged_table = count_table.astype(str) + '(' + rate_table_fmt + ')'

            with pd.option_context(
                'display.max_rows', None,
                'display.max_columns', None,
                'display.width', 2000,
                'display.expand_frame_repr', False
            ):
                print(merged_table)
            print()

    print('=' * 80)
    print()


# 이진분류 모델의 성능을 출력하고 주요 지표를 표로 반환하는 함수
def evaluate_binary_classifier(model, X, y, data_name: str = 'train', threshold: float = 0.5) -> pd.DataFrame:
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    auc = roc_auc_score(y, y_prob)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, zero_division=0)

    result_table = pd.DataFrame([{
        'data': data_name,
        'threshold': threshold,
        'accuracy': round(acc, 4),
        'precision_1': round(prec, 4),
        'recall_1': round(rec, 4),
        'f1_1': round(f1, 4),
        'roc_auc': round(auc, 4)
    }])

    print(f'[{data_name} Model Evaluation]')
    print('=' * 50)
    print(f'threshold: {threshold}')
    print('-' * 50)
    print(result_table)
    print()

    print('[Confusion Matrix]')
    print('=' * 50)
    print(cm)
    print()

    print('[Classification Report]')
    print('=' * 50)
    print(report)
    print()

    print('[Score Summary]')
    print('=' * 50)
    print(f'accuracy : {acc:.4f}')
    print(f'precision: {prec:.4f}')
    print(f'recall   : {rec:.4f}')
    print(f'f1-score : {f1:.4f}')
    print(f'roc-auc  : {auc:.4f}')
    print()
    return result_table


# 스레드 홀드 결정 함수
def make_threshold_table(
    model,
    X,
    y,
    data_name: str = 'valid',
    thresholds = np.arange(0.1, 0.91, 0.05)
) -> pd.DataFrame:

    y_prob = model.predict_proba(X)[:, 1]
    result_list = []

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)

        result_list.append({
            'data': data_name,
            'threshold': round(float(th), 3),
            'accuracy': round(accuracy_score(y, y_pred), 4),
            'precision_1': round(precision_score(y, y_pred, zero_division=0), 4),
            'recall_1': round(recall_score(y, y_pred, zero_division=0), 4),
            'f1_1': round(f1_score(y, y_pred, zero_division=0), 4),
            'roc_auc': round(roc_auc_score(y, y_prob), 4)
        })

    thresh_table = pd.DataFrame(result_list)

    print(f'[{data_name} Threshold Check]')
    print('=' * 50)
    print('threshold별 성능 확인')
    print('-' * 50)
    print(thresh_table)
    print()

    best_f1_row = thresh_table.sort_values(['f1_1', 'recall_1', 'precision_1'], ascending=False).iloc[0]
    best_recall_row = thresh_table.sort_values(['recall_1', 'precision_1', 'f1_1'], ascending=False).iloc[0]

    print('[Threshold Summary]')
    print('=' * 50)
    print('f1 기준 best threshold')
    print('-' * 50)
    print(best_f1_row)
    print()

    print('recall 기준 best threshold')
    print('-' * 50)
    print(best_recall_row)
    print()

    return thresh_table


# precision 최소 기준을 만족하는 threshold 중 recall이 가장 높은 threshold를 찾는 함수
def select_threshold_by_recall(
    thresh_table: pd.DataFrame,
    min_precision: float = 0.5
) -> pd.DataFrame:

    filtered = thresh_table[thresh_table['precision_1'] >= min_precision].copy()

    print('[Threshold Selection]')
    print('=' * 50)
    print(f'min precision 기준: {min_precision}')
    print('-' * 50)

    if filtered.empty:
        print('조건을 만족하는 threshold가 없습니다.')
        print()
        return filtered

    selected = filtered.sort_values(
        ['recall_1', 'f1_1', 'precision_1'],
        ascending=False
    ).head(1)

    print('선택된 threshold')
    print('-' * 50)
    print(selected)
    print()

    return selected

#---------------------------------[변수해석]---------------------------------------#

# 수치 연속형 변수 pdp 그리는 함수
    #사용방법 plot_pdp_with_hist(모델, 입력, 수치특성, task=회귀/분류)
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence

def make_numeric_pdp(
    model,
    X,
    features,
    task="classification",              # "classification" or "regression"
    class_index=1,                      # classification에서 보고 싶은 class index
    response_method=None,               # None이면 자동 결정
    grid_resolution=50,
    percentiles=(0.05, 0.95),
    bins=30,
    n_cols=2,
    figsize_per_panel=(9, 6), #7.2, 4.8
    title=None,
    show_rug=True,
    threshold_line=0.45,
    wspace=0.32,
    hspace=0.38,
    return_individual=True, dpi=240              # True면 individual(ICE 원자료)도 반환
):
    X_plot = X.copy()
    features = list(features)
    n_features = len(features)
    results = {}

    if response_method is None:
        if task == "classification":
            if hasattr(model, "predict_proba"):
                response_method = "predict_proba"
            elif hasattr(model, "decision_function"):
                response_method = "decision_function"
            else:
                response_method = "auto"
        else:
            response_method = "auto"

    kind = "both" if return_individual else "average"

    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False
    )
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]

        pd_result = partial_dependence(
            estimator=model,
            X=X_plot,
            features=[feat],
            response_method=response_method,
            grid_resolution=grid_resolution,
            percentiles=percentiles,
            kind=kind,
            method="brute"
        )

        x_vals = pd_result["grid_values"][0] if "grid_values" in pd_result else pd_result["values"][0]

        avg = np.asarray(pd_result["average"])
        if task == "classification":
            if avg.ndim == 2:
                y_vals = avg[class_index] if avg.shape[0] > 1 else avg[0]
            elif avg.ndim == 1:
                y_vals = avg
            else:
                y_vals = np.squeeze(avg)
                if y_vals.ndim > 1:
                    y_vals = y_vals[class_index]
        else:
            if avg.ndim == 2:
                y_vals = avg[0]
            else:
                y_vals = np.squeeze(avg)

        # individual(ICE 원자료)
        individual_vals = None
        pdp_std = None
        if return_individual and "individual" in pd_result:
            ind = np.asarray(pd_result["individual"])

            if task == "classification":
                if ind.ndim == 3:
                    individual_vals = ind[class_index] if ind.shape[0] > 1 else ind[0]
                elif ind.ndim == 2:
                    individual_vals = ind
                else:
                    individual_vals = np.squeeze(ind)
                    if individual_vals.ndim > 2:
                        individual_vals = individual_vals[class_index]
            else:
                if ind.ndim == 3:
                    individual_vals = ind[0]
                else:
                    individual_vals = np.squeeze(ind)

            if individual_vals is not None:
                pdp_std = np.std(individual_vals, axis=0, ddof=1) if individual_vals.shape[0] > 1 else np.full_like(y_vals, np.nan, dtype=float)

        # pdp_table
        pdp_table = pd.DataFrame({
            "feature_value": x_vals,
            "pdp_mean": y_vals
        })
        pdp_table["diff"] = pdp_table["pdp_mean"].diff().round(4)
        pdp_table.loc[pdp_table['diff'].abs() < 1e-10, "diff"] = 0
        pdp_table["slope"] = (pdp_table["pdp_mean"].diff() / pdp_table["feature_value"].diff()).round(4)
        pdp_table.loc[pdp_table["slope"].abs() < 1e-10, "slope"] = 0

        pdp_table["abs_slope"] = pdp_table["slope"].abs().round(4)

        if pdp_std is not None:
            pdp_table["pdp_std"] = pdp_std
            pdp_table = pdp_table[["feature_value", "pdp_mean", "pdp_std", "diff", "slope", "abs_slope"]]

        # 그래프
        ax.plot(x_vals, y_vals, linewidth=2, label="PDP Mean")

        if (
            task == "classification"
            and response_method == "predict_proba"
            and threshold_line is not None
        ):
            ax.axhline(
                threshold_line,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=f"threshold={threshold_line:.2f}"
            )

        ax_hist = ax.twinx()
        ax_hist.hist(
            X_plot[feat].dropna().values,
            bins=bins,
            alpha=0.22
        )
        ax_hist.set_ylabel("Count", fontsize=9)
        ax_hist.grid(False)

        if show_rug:
            x_nonnull = X_plot[feat].dropna().values
            if len(x_nonnull) > 0:
                ymin, ymax = ax.get_ylim()
                rug_y = ymin + (ymax - ymin) * 0.02
                ax.plot(
                    x_nonnull,
                    np.full_like(x_nonnull, rug_y, dtype=float),
                    "|",
                    markersize=6,
                    alpha=0.20
                )

        clean_feat = feat.split("__", 1)[1] if "__" in feat else feat

        ax.set_title(clean_feat, fontsize=12, pad=10)
        ax.set_xlabel(clean_feat)

        if task == "classification":
            if response_method == "predict_proba":
                ax.set_ylabel("Predicted probability")
            elif response_method == "decision_function":
                ax.set_ylabel("Decision score")
            else:
                ax.set_ylabel("Model response")
        else:
            ax.set_ylabel("Predicted value")

        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.25)

        results[clean_feat] = {
            "feature": clean_feat,
            "grid_values": np.asarray(x_vals),
            "average": np.asarray(y_vals),
            "individual": individual_vals,
            "pdp_table": pdp_table
        }

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if title is not None:
        fig.suptitle(title, fontsize=15, y=0.98)

    fig.subplots_adjust(wspace=wspace, hspace=hspace, top=0.90 if title is not None else 0.95)
    plt.show()

    return results
    

# 범주 명목형 변수 PDP & 관측평균확률표
    # 함수 사용법 : 함수(모델, 입력, 칼럼, task=분류/회귀)
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

def make_categorical_pdp_group(
    estimator,
    X,
    columns,
    task="classification",              # "classification" or "regression"
    class_index=1,                      # classification에서 보고 싶은 class index
    response_method=None,               # None이면 자동 결정
    n_cols=2,
    figsize_per_panel=(9, 6),
    threshold_line=0.45,
    title=None,
    sort_levels=False,
    show_rug=True,
    wspace=0.32,
    hspace=0.42,
    xtick_rotation=30, dpi=240
):
    X = X.copy()
    columns = list(columns)
    results = {}

    if response_method is None:
        if task == "classification":
            if hasattr(estimator, "predict_proba"):
                response_method = "predict_proba"
            elif hasattr(estimator, "decision_function"):
                response_method = "decision_function"
            else:
                response_method = "predict"
        else:
            response_method = "predict"

    def _predict_response(model, X_input):
        if task == "classification":
            if response_method == "predict_proba" and hasattr(model, "predict_proba"):
                pred = np.asarray(model.predict_proba(X_input))
                return pred[:, class_index] if pred.ndim == 2 else pred
            elif response_method == "decision_function" and hasattr(model, "decision_function"):
                return np.asarray(model.decision_function(X_input))
            else:
                return np.asarray(model.predict(X_input))
        else:
            return np.asarray(model.predict(X_input))

    groups = OrderedDict()
    used_cols = [c for c in columns if c in X.columns]

    cat_cols = [c for c in used_cols if c.startswith("cat__")]
    stripped_cat_cols = [c[len("cat__"):] for c in cat_cols]
    group_map = OrderedDict()

    # 원-핫 그룹 자동 추론
    for full_col, stripped in zip(cat_cols, stripped_cat_cols):
        underscore_positions = [i for i, ch in enumerate(stripped) if ch == "_"]
        chosen_prefix = None

        for pos in underscore_positions:
            candidate = stripped[:pos]
            matched = [s for s in stripped_cat_cols if s.startswith(candidate + "_")]
            if len(matched) >= 2:
                chosen_prefix = candidate

        if chosen_prefix is None:
            chosen_prefix = stripped

        level = stripped if stripped == chosen_prefix else stripped[len(chosen_prefix) + 1:]
        group_map.setdefault(chosen_prefix, []).append((full_col, level))

    for group_name, items in group_map.items():
        groups[group_name] = {
            "type": "onehot",
            "columns": [col for col, _ in items],
            "levels": [lvl for _, lvl in items]
        }

    # 비-cat__ 이진변수 자동 추론
    non_cat_cols = [c for c in used_cols if not c.startswith("cat__")]
    for col in non_cat_cols:
        nunique = X[col].dropna().nunique()
        if nunique <= 2:
            clean_name = col.split("__", 1)[1] if "__" in col else col
            levels = sorted(pd.Series(X[col].dropna().unique()).tolist())
            groups[clean_name] = {
                "type": "binary",
                "columns": [col],
                "levels": levels
            }

    if len(groups) == 0:
        raise ValueError("입력한 columns 중 자동 추론된 범주형/이진변수가 없습니다.")

    original_pred = _predict_response(estimator, X)
    n = len(X)

    rows = []

    for feature_name, info in groups.items():
        if info["type"] == "onehot":
            group_cols = info["columns"]

            for col, level in zip(info["columns"], info["levels"]):
                mask = (X[col] == 1)
                count = int(mask.sum())
                observed_mean = original_pred[mask].mean() if count > 0 else np.nan
                observed_std = original_pred[mask].std(ddof=1) if count > 1 else np.nan

                X_temp = X.copy()
                X_temp[group_cols] = 0
                X_temp[col] = 1
                forced_pred = _predict_response(estimator, X_temp)
                pdp_mean = forced_pred.mean()
                pdp_std = forced_pred.std(ddof=1) if len(forced_pred) > 1 else np.nan

                rows.append({
                    "feature": feature_name,
                    "label": level,
                    "column": col,
                    "count": count,
                    "proportion": f"{round((count / n) * 100, 3)}%",
                    "observed_mean": observed_mean,
                    "observed_std": observed_std,
                    "pdp_mean": pdp_mean,
                    "pdp_std": pdp_std,
                    "mean_diff": pdp_mean - observed_mean
                })

        elif info["type"] == "binary":
            col = info["columns"][0]

            for level in info["levels"]:
                mask = (X[col] == level)
                count = int(mask.sum())
                observed_mean = original_pred[mask].mean() if count > 0 else np.nan
                observed_std = original_pred[mask].std(ddof=1) if count > 1 else np.nan

                X_temp = X.copy()
                X_temp[col] = level
                forced_pred = _predict_response(estimator, X_temp)
                pdp_mean = forced_pred.mean()
                pdp_std = forced_pred.std(ddof=1) if len(forced_pred) > 1 else np.nan

                rows.append({
                    "feature": feature_name,
                    "label": level,
                    "column": col,
                    "count": count,
                    "proportion": f"{round((count / n) * 100, 3)}%",
                    "observed_mean": observed_mean,
                    "observed_std": observed_std,
                    "pdp_mean": pdp_mean,
                    "pdp_std": pdp_std,
                    "mean_diff": pdp_mean - observed_mean
                })

    pdp_table_all = (
        pd.DataFrame(rows)
        .sort_values(["feature", "label"])
        .reset_index(drop=True)
    )

    feature_names = list(groups.keys())
    n_features = len(feature_names)
    n_rows = math.ceil(n_features / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False
    )
    axes = axes.flatten()

    for i, feat in enumerate(feature_names):
        ax = axes[i]
        sub = pdp_table_all[pdp_table_all["feature"] == feat].copy()

        if sort_levels:
            sub = sub.sort_values("pdp_mean", ascending=True)

        x_labels = sub["label"].astype(str).tolist()
        x_pos = np.arange(len(sub))

        ax.plot(
            x_pos,
            sub["pdp_mean"].values,
            marker="o",
            linewidth=2,
            label="PDP Mean"
        )

        ax.plot(
            x_pos,
            sub["observed_mean"].values,
            marker="s",
            linewidth=1.6,
            alpha=0.9,
            label="Observed Mean"
        )

        if (
            task == "classification"
            and response_method == "predict_proba"
            and threshold_line is not None
        ):
            ax.axhline(
                threshold_line,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=f"threshold={threshold_line:.2f}"
            )

        ax_hist = ax.twinx()
        ax_hist.bar(
            x_pos,
            sub["count"].values,
            alpha=0.22,
            width=0.65
        )
        ax_hist.set_ylabel("Count", fontsize=9)
        ax_hist.grid(False)

        if show_rug:
            rug_x = np.repeat(x_pos, sub["count"].astype(int).values)
            if len(rug_x) > 0:
                ymin, ymax = ax.get_ylim()
                rug_y = ymin + (ymax - ymin) * 0.02
                ax.plot(
                    rug_x,
                    np.full_like(rug_x, rug_y, dtype=float),
                    "|",
                    markersize=6,
                    alpha=0.20
                )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=xtick_rotation, ha="right")
        ax.set_title(feat, fontsize=12, pad=10)
        ax.set_xlabel(feat)

        if task == "classification":
            if response_method == "predict_proba":
                ax.set_ylabel("Predicted probability")
            elif response_method == "decision_function":
                ax.set_ylabel("Decision score")
            else:
                ax.set_ylabel("Model response")
        else:
            ax.set_ylabel("Predicted value")

        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)

        results[feat] = {
            "feature": feat,
            "group_cols": groups[feat]["columns"],
            "labels": sub["label"].tolist(),
            "average": sub["pdp_mean"].to_numpy(),
            "pdp_table": sub.reset_index(drop=True)
        }

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if title is not None:
        fig.suptitle(title, fontsize=15, y=0.98)

    fig.subplots_adjust(wspace=wspace, hspace=hspace, top=0.88 if title is not None else 0.95)
    plt.show()

    return results


# 전역 SHAP 평균 표, 막대그래프, beesworm 그래프 그리는 함수
    #사용방법: plot_shap_global(테스트 입력, SHAP 밸류, 특성 리스트, 카테고리 리스트-필요시)
def plot_shap_global(
    X,
    shap_values,
    selected_features=None,
    categorical_bases=None,
    max_display=15,
    title_bar="Grouped SHAP Importance",
    title_beeswarm="Grouped SHAP Beeswarm",
    display_table=True,
    show_bar=True,
    show_beeswarm=True,
    bar_figsize=(8.8, 4.8),
    beeswarm_plot_size=(8.8, 5.0),
):
    def _make_grouped(shap_values, X, meta):
        shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
        grouped_shap = pd.DataFrame(index=X.index)
        grouped_value = pd.DataFrame(index=X.index)
        grouped_type = {}

        for g, sub in meta.groupby("group", sort=False):
            cols = sub["encoded_col"].tolist()
            gtype = sub["group_type"].iloc[0]
            grouped_type[g] = gtype

            grouped_shap[g] = shap_df[cols].sum(axis=1)

            if gtype == "numeric":
                grouped_value[g] = pd.to_numeric(X[cols[0]], errors="coerce")
            else:
                levels = sub["level"].fillna(g).tolist()
                arr = X[cols].to_numpy()
                max_idx = arr.argmax(axis=1)
                max_val = arr.max(axis=1)
                labels = np.array(levels, dtype=object)[max_idx]
                labels = np.where(max_val > 0.5, labels, "None")
                grouped_value[g] = pd.Categorical(labels).codes.astype(float)

        return grouped_shap, grouped_value, grouped_type

    def _resolve_selected_groups(selected_features, meta, available_groups):
        if selected_features is None:
            return None

        encoded_to_group = dict(zip(meta["encoded_col"], meta["group"]))
        resolved = []
        missing = []

        for feat in selected_features:
            if feat in available_groups:
                resolved.append(feat)
            elif feat in encoded_to_group:
                resolved.append(encoded_to_group[feat])
            elif f"remainder__{feat}" in encoded_to_group:
                resolved.append(encoded_to_group[f"remainder__{feat}"])
            else:
                cat_matches = meta.loc[meta["group"] == feat, "group"].tolist()
                if cat_matches:
                    resolved.append(feat)
                else:
                    missing.append(feat)

        resolved = list(dict.fromkeys(resolved))
        if missing:
            print("Ignoring missing features:", missing)
        return resolved

    if np.asarray(shap_values).ndim != 2:
        raise ValueError("plot_shap_global requires 2D shap_values.")
    if X.shape != np.asarray(shap_values).shape:
        raise ValueError(f"X.shape={X.shape}, shap_values.shape={np.asarray(shap_values).shape} must match.")

    meta = _build_meta(X.columns, categorical_bases)
    grouped_shap, grouped_value, grouped_type = _make_grouped(shap_values, X, meta)

    resolved_groups = _resolve_selected_groups(
        selected_features, meta, available_groups=grouped_shap.columns.tolist()
    )

    if resolved_groups is not None:
        grouped_shap = grouped_shap[resolved_groups]
        grouped_value = grouped_value[resolved_groups]
        grouped_type = {k: v for k, v in grouped_type.items() if k in resolved_groups}

    tbl = pd.DataFrame(
        {
            "mean_abs_shap": grouped_shap.abs().mean(),
            "mean_shap": grouped_shap.mean(),
            "std_shap": grouped_shap.std(),
        }
    )
    tbl["feature_type"] = [grouped_type[c] for c in tbl.index]
    tbl = tbl.sort_values("mean_abs_shap", ascending=False).head(max_display)
    tbl_rounded = tbl.round(5)

    if display_table:
        display(tbl_rounded)

    top_features = tbl.index.tolist()

    if show_bar:
        h = max(bar_figsize[1], len(top_features) * 0.32)
        fig, ax = plt.subplots(figsize=(bar_figsize[0], h))
        y_pos = np.arange(len(top_features))

        ax.barh(
            y_pos,
            tbl.loc[top_features, "mean_abs_shap"].values,
            color=SHAP_BAR_COLOR,
            alpha=0.84,
            height=0.58
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features, fontsize=10, color="black")
        ax.invert_yaxis()
        ax.set_xlabel("mean(|SHAP value|)", fontsize=10.5, color="black")
        ax.set_title(title_bar, fontsize=12.5, pad=8, color="#111111")
        _style_axis(ax, ygrid=True, xgrid=True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

        plt.tight_layout()
        plt.show()

    if show_beeswarm:
        plot_cmap = _make_shap_cmap()
        plot_h = max(beeswarm_plot_size[1], len(top_features) * 0.34)

        shap.summary_plot(
            grouped_shap[top_features].to_numpy(),
            features=grouped_value[top_features],
            feature_names=top_features,
            plot_type="dot",
            max_display=len(top_features),
            show=False,
            color=plot_cmap,
            plot_size=(beeswarm_plot_size[0], plot_h)
        )

        ax = plt.gca()
        fig = plt.gcf()

        ax.set_title(title_beeswarm, fontsize=12.5, pad=8, color="#111111")
        ax.set_xlabel("SHAP Value", fontsize=10.5, color="black")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=10, colors="black")
        ax.tick_params(axis="y", labelsize=10, colors="black", length=0)

        _style_axis(ax, ygrid=True, xgrid=True)
        ax.spines["left"].set_visible(False)

        for other_ax in fig.axes:
            if other_ax is not ax:
                other_ax.set_ylabel("")
                other_ax.tick_params(labelsize=10, colors="black")

        plt.tight_layout()
        plt.show()

    result = {
        "importance_table": tbl_rounded,
        "grouped_shap": grouped_shap,
        "grouped_value": grouped_value,
        "grouped_type": grouped_type,
        "top_features": top_features,
    }

    return result


# SHAP dependence plot 그리는 함수
    #사용방법: plot_shap_dependence(테스트 입력값, shap값, 특성(리스트), 카테고리 리스트)
def plot_shap_dependence(
    X,
    shap_values,
    feature,
    categorical_bases=None,
    interaction_index="auto",
    title_prefix="SHAP DP",
    point_alpha=0.70,
    point_size=20,
    n_cols=2,
    figsize_per_panel=(5.8, 4.2),
    title=None,
    wspace=0.55,
    hspace=0.70,
):
    def _encoded_to_display(col_name, meta):
        hit = meta[meta["encoded_col"] == col_name]
        if len(hit) == 0:
            return col_name
        row = hit.iloc[0]
        if row["group_type"] == "categorical":
            level = row.get("level", None)
            return f"{row['group']} = {level}" if level is not None else row["group"]
        return row["group"]

    def _plot_one_feature(ax, feature_name):
        meta = _build_meta(X.columns, categorical_bases)
        feature_group = _resolve_feature_to_group(feature_name, meta)
        sub = meta[meta["group"] == feature_group].copy()
        shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)

        if len(sub) == 0:
            raise ValueError(f"Feature '{feature_name}' not found.")

        gtype = sub["group_type"].iloc[0]

        if gtype == "numeric":
            encoded_col = sub["encoded_col"].iloc[0]
            x_main = pd.to_numeric(X[encoded_col], errors="coerce").to_numpy(dtype=float)
            y_main = pd.to_numeric(shap_df[encoded_col], errors="coerce").to_numpy(dtype=float)

            if interaction_index == "auto":
                inter_col = _best_interaction_feature(x_main, y_main, X, exclude_cols={encoded_col})
            elif interaction_index is None:
                inter_col = None
            else:
                inter_col = _resolve_feature_to_encoded(interaction_index, meta, X.columns)

            title_label = feature_group
            xlabel_label = feature_group

        else:
            row = sub.iloc[0]
            encoded_col = row["encoded_col"]
            level = row["level"]
            x_main = pd.to_numeric(X[encoded_col], errors="coerce").to_numpy(dtype=float)
            y_main = pd.to_numeric(shap_df[encoded_col], errors="coerce").to_numpy(dtype=float)

            if interaction_index == "auto":
                inter_col = _best_interaction_feature(x_main, y_main, X, exclude_cols={encoded_col})
            elif interaction_index is None:
                inter_col = None
            else:
                inter_col = _resolve_feature_to_encoded(interaction_index, meta, X.columns)

            title_label = f"{feature_group} = {level}" if level is not None else feature_group
            xlabel_label = title_label

        order = _density_order(x_main, y_main)
        x_plot = x_main[order]
        y_plot = y_main[order]

        cmap = _make_shap_cmap()

        if inter_col is None:
            ax.scatter(
                x_plot, y_plot,
                s=point_size,
                alpha=point_alpha,
                color="#7B6FD0",
                edgecolors="none",
                linewidths=0
            )
            c_norm = None
            c_raw = None
        else:
            c_raw, c_norm = _get_color_values_and_norm(X[inter_col], q_low=5, q_high=95)
            c_plot = c_raw[order]
            ax.scatter(
                x_plot, y_plot,
                c=c_plot,
                cmap=cmap,
                norm=c_norm,
                s=point_size,
                alpha=point_alpha,
                edgecolors="none",
                linewidths=0
            )

        ax.axhline(0, linestyle="--", linewidth=1.0, color="#98A2B3", alpha=0.85)
        ax.set_title(f"{title_prefix} - {title_label}", fontsize=11, pad=9, color="#111111")
        ax.set_xlabel(xlabel_label, fontsize=10, color="black")
        ax.set_ylabel("SHAP Value", fontsize=10, color="black")
        _style_axis(ax, ygrid=True, xgrid=True)

        if inter_col is not None and c_norm is not None:
            sm = ScalarMappable(norm=c_norm, cmap=cmap)
            sm.set_array([])

            cbar = plt.colorbar(
                sm,
                ax=ax,
                pad=0.04,
                fraction=0.06
            )
            cbar.set_ticks([c_norm.vmin, c_norm.vmax])
            cbar.set_ticklabels(["Low", "High"])
            cbar.ax.tick_params(labelsize=9, colors="black")
            cbar.outline.set_visible(False)
            cbar.set_label("")

            inter_label = _encoded_to_display(inter_col, meta)
            cbar.ax.set_ylabel(
                f"interaction:\n{inter_label}",
                rotation=270,
                labelpad=1,
                fontsize=8,
                color="black",
                va="bottom"
            )
        else:
            inter_label = None

        dependence_table = pd.DataFrame({
            "feature_value": x_main,
            "shap_value": y_main,
            "abs_shap": np.abs(y_main)
        })

        if inter_col is not None and c_raw is not None:
            dependence_table["interaction_value"] = c_raw
            dependence_table["interaction_feature"] = inter_label
        else:
            dependence_table["interaction_value"] = np.nan
            dependence_table["interaction_feature"] = None

        result = {
            "feature": feature_group,
            "display_feature": title_label,
            "encoded_col": encoded_col,
            "group_type": gtype,
            "interaction_feature": inter_label,
            "dependence_table": dependence_table
        }

        return result

    if np.asarray(shap_values).ndim != 2:
        raise ValueError("plot_shap_dependence requires 2D shap_values.")
    if X.shape != np.asarray(shap_values).shape:
        raise ValueError(f"X.shape={X.shape}, shap_values.shape={np.asarray(shap_values).shape} must match.")

    features = [feature] if isinstance(feature, str) else list(feature)
    n_features = len(features)
    results = {}

    n_rows = math.ceil(n_features / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        squeeze=False
    )
    axes = axes.flatten()

    for i, feat in enumerate(features):
        results[feat] = _plot_one_feature(axes[i], feat)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if title is not None:
        fig.suptitle(title, fontsize=12.5, y=0.98, color="#111111")

    fig.subplots_adjust(
        wspace=wspace,
        hspace=hspace,
        top=0.90 if title is not None else 0.94
    )
    plt.show()

    return results


# SHAP water fall 그리는 함수
    #사용방법: plot_shap_waterfall(테스트 입력값, shap값, 예상값, 행선택)
def plot_shap_waterfall(
    X,
    shap_values,
    expected_value,
    row=0,
    selected_features=None,
    categorical_bases=None,
    grouped=True,
    max_display=12,
    title_prefix="Waterfall",
    figsize=(8.6, 5.2),
    title_fontsize=12.5,
    tick_fontsize=9.5,
):
    def _make_grouped(shap_values, X, meta):
        shap_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
        grouped_shap = pd.DataFrame(index=X.index)
        grouped_value = pd.DataFrame(index=X.index)
        grouped_type = {}

        for g, sub in meta.groupby("group", sort=False):
            cols = sub["encoded_col"].tolist()
            gtype = sub["group_type"].iloc[0]
            grouped_type[g] = gtype
            grouped_shap[g] = shap_df[cols].sum(axis=1)

            if gtype == "numeric":
                grouped_value[g] = X[cols[0]]
            else:
                levels = sub["level"].fillna(g).tolist()
                arr = X[cols].to_numpy()
                max_idx = arr.argmax(axis=1)
                max_val = arr.max(axis=1)
                labels = np.array(levels, dtype=object)[max_idx]
                labels = np.where(max_val > 0.5, labels, "None")
                grouped_value[g] = labels

        return grouped_shap, grouped_value, grouped_type

    def _resolve_selected_groups(selected_features, meta, available_groups):
        if selected_features is None:
            return None

        encoded_to_group = dict(zip(meta["encoded_col"], meta["group"]))
        resolved, missing = [], []

        for feat in selected_features:
            if feat in available_groups:
                resolved.append(feat)
            elif feat in encoded_to_group:
                resolved.append(encoded_to_group[feat])
            elif f"remainder__{feat}" in encoded_to_group:
                resolved.append(encoded_to_group[f"remainder__{feat}"])
            else:
                cat_matches = meta.loc[meta["group"] == feat, "group"].tolist()
                if cat_matches:
                    resolved.append(feat)
                else:
                    missing.append(feat)

        resolved = list(dict.fromkeys(resolved))
        if missing:
            print("Ignoring missing features:", missing)
        return resolved

    def _resolve_selected_encoded(selected_features, X_cols, meta):
        if selected_features is None:
            return None

        encoded_cols, missing = [], []
        for feat in selected_features:
            if feat in X_cols:
                encoded_cols.append(feat)
            elif f"remainder__{feat}" in X_cols:
                encoded_cols.append(f"remainder__{feat}")
            else:
                cat_cols = meta.loc[meta["group"] == feat, "encoded_col"].tolist()
                if cat_cols:
                    encoded_cols.extend(cat_cols)
                else:
                    missing.append(feat)

        encoded_cols = list(dict.fromkeys(encoded_cols))
        if missing:
            print("Ignoring missing features:", missing)
        return encoded_cols

    def _to_2d_shap(shap_values, X):
        arr = np.asarray(shap_values)
        if arr.ndim == 1:
            if arr.shape[0] != X.shape[1]:
                raise ValueError(f"1D shap_values length {arr.shape[0]} != number of columns {X.shape[1]}.")
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError("shap_values must be 1D or 2D.")
        if arr.shape[1] != X.shape[1]:
            raise ValueError(f"shap_values.shape[1]={arr.shape[1]} must equal X.shape[1]={X.shape[1]}.")
        return arr

    def _fix_waterfall_texts(ax):
        x0, x1 = ax.get_xlim()
        xr = x1 - x0
        left_bound = x0 + xr * 0.03
        right_bound = x1 - xr * 0.03

        for txt in list(ax.texts):
            s = txt.get_text()
            if not s or s.strip() == "":
                continue

            is_value_text = any(ch.isdigit() for ch in s) or ("−" in s) or ("-" in s) or ("+" in s)
            if not is_value_text:
                txt.set_color("#111111")
                txt.set_fontsize(tick_fontsize)
                continue

            x, y = txt.get_position()
            if x < left_bound:
                x = left_bound
                txt.set_ha("left")
            elif x > right_bound:
                x = right_bound
                txt.set_ha("right")

            txt.set_position((x, y))
            txt.set_clip_on(False)
            txt.set_color("#111111")
            txt.set_fontsize(tick_fontsize - 0.3)
            txt.set_bbox(
                dict(
                    boxstyle="round,pad=0.16",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.86,
                )
            )

    shap_arr = _to_2d_shap(shap_values, X)

    if shap_arr.shape[0] == X.shape[0]:
        X_row = X.iloc[[row]].copy()
        shap_row = shap_arr[[row]]
    elif shap_arr.shape[0] == 1:
        X_row = X.iloc[[row]].copy()
        shap_row = shap_arr
    else:
        raise ValueError("For waterfall, shap_values must be full 2D array or single-row 1D/2D array.")

    meta = _build_meta(X.columns, categorical_bases)
    base_value = float(np.asarray(expected_value).reshape(-1)[0])

    if grouped:
        grouped_shap, grouped_value, grouped_type = _make_grouped(shap_row, X_row, meta)

        resolved_groups = _resolve_selected_groups(
            selected_features, meta, available_groups=grouped_shap.columns.tolist()
        )
        if resolved_groups is not None:
            grouped_shap = grouped_shap[resolved_groups]
            grouped_value = grouped_value[resolved_groups]
            grouped_type = {k: v for k, v in grouped_type.items() if k in resolved_groups}

        vals = grouped_shap.iloc[0]
        data_vals = []
        feature_names = []

        for feat in vals.index:
            data_vals.append(grouped_value.iloc[0][feat])
            feature_names.append(feat)

        order = np.argsort(np.abs(vals.to_numpy()))[::-1]
        vals = vals.iloc[order]
        feature_names = [feature_names[i] for i in order]
        data_vals = [data_vals[i] for i in order]

        exp = shap.Explanation(
            values=vals.to_numpy(),
            base_values=expected_value,
            data=np.array(data_vals, dtype=object),
            feature_names=feature_names,
        )

        waterfall_table = pd.DataFrame({
            "feature": feature_names,
            "feature_value": data_vals,
            "shap_value": vals.to_numpy(),
            "abs_shap": np.abs(vals.to_numpy())
        })

        prediction = base_value + vals.to_numpy().sum()

        plt.figure(figsize=figsize)
        shap.plots.waterfall(exp, max_display=max_display, show=False)
        ax = plt.gca()

        ax.set_title(f"{title_prefix} (grouped, row={row})", fontsize=title_fontsize, pad=8, color="#111111")
        _style_axis(ax, ygrid=True, xgrid=True)
        _fix_waterfall_texts(ax)

        plt.tight_layout()
        plt.show()

        result = {
            "row": row,
            "grouped": True,
            "base_value": base_value,
            "prediction": prediction,
            "waterfall_table": waterfall_table
        }

        return result

    else:
        X_use = X.iloc[[row]].copy()
        shap_use = pd.DataFrame(shap_row, columns=X.columns, index=X_use.index)

        resolved_encoded = _resolve_selected_encoded(
            selected_features, X_cols=X_use.columns.tolist(), meta=meta
        )
        if resolved_encoded is not None:
            X_use = X_use[resolved_encoded]
            shap_use = shap_use[resolved_encoded]

        vals = shap_use.iloc[0]
        order = np.argsort(np.abs(vals.to_numpy()))[::-1]
        vals = vals.iloc[order]
        X_vals = X_use.iloc[0][vals.index]

        exp = shap.Explanation(
            values=vals.to_numpy(),
            base_values=expected_value,
            data=X_vals.to_numpy(),
            feature_names=[_pretty_name(c) for c in vals.index],
        )

        waterfall_table = pd.DataFrame({
            "feature": [_pretty_name(c) for c in vals.index],
            "feature_value": X_vals.to_numpy(),
            "shap_value": vals.to_numpy(),
            "abs_shap": np.abs(vals.to_numpy())
        })

        prediction = base_value + vals.to_numpy().sum()

        plt.figure(figsize=figsize)
        shap.plots.waterfall(exp, max_display=max_display, show=False)
        ax = plt.gca()

        ax.set_title(f"{title_prefix} (encoded, row={row})", fontsize=title_fontsize, pad=8, color="#111111")
        _style_axis(ax, ygrid=True, xgrid=True)
        _fix_waterfall_texts(ax)

        plt.tight_layout()
        plt.show()

        result = {
            "row": row,
            "grouped": False,
            "base_value": base_value,
            "prediction": prediction,
            "waterfall_table": waterfall_table
        }

        return result


# 데이터프레임 결과를 csv와 xlsx 파일로 저장하는 함수
def save_table(
    df: pd.DataFrame,
    file_name: str,
    folder: str = 'outputs',
    index: bool = True
) -> None:

    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    csv_path = folder_path / f'{file_name}.csv'
    xlsx_path = folder_path / f'{file_name}.xlsx'

    df.to_csv(csv_path, index=index, encoding='utf-8-sig')
    df.to_excel(xlsx_path, index=index)

    print('[Save Table]')
    print('=' * 50)
    print(f'file name: {file_name}')
    print('-' * 50)
    print(f'csv saved : {csv_path}')
    print(f'xlsx saved: {xlsx_path}')
    print()
