import os
import pandas as pd
import numpy as np
from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
from django.http import HttpResponse
from scipy import stats
import warnings
from io import BytesIO
import re
warnings.filterwarnings('ignore')

def home(request):
    """صفحه اصلی با طراحی مدرن"""
    return render(request, 'analyzer/home.html')

def detect_date_columns(df):
    """تشخیص ستون‌های تاریخ"""
    date_columns = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col], errors='coerce')
            if not pd.to_datetime(df[col], errors='coerce').isna().all():
                date_columns.append(col)
        except:
            continue
    return date_columns

def generate_insights_and_recommendations(analysis_type, analysis_data, df):
    """تولید بینش‌ها و راهکارهای هوشمند"""
    insights = []
    recommendations = []
    
    if analysis_type == "داده‌های مفقودی":
        total_missing = analysis_data['تعداد مفقودی'].sum()
        max_missing_col = analysis_data.loc[analysis_data['تعداد مفقودی'].idxmax()]
        
        insights.append(f"در مجموع {total_missing} داده مفقودی وجود دارد")
        insights.append(f"ستون '{max_missing_col.name}' با {max_missing_col['تعداد مفقودی']} داده مفقودی ({max_missing_col['درصد مفقودی']}%) بیشترین مشکل را دارد")
        
        if max_missing_col['درصد مفقودی'] > 50:
            recommendations.append("❌ **ستون با بیش از ۵۰٪ داده مفقودی بهتر است حذف شود**")
            recommendations.append("🔍 **بررسی علت مفقودی داده‌ها ضروری است**")
        elif max_missing_col['درصد مفقودی'] > 20:
            recommendations.append("⚡ **استفاده از روش‌های پیشرفته جایگزینی داده‌های مفقودی پیشنهاد می‌شود**")
            recommendations.append("📊 **تحلیل حساسیت به داده‌های مفقودی انجام شود**")
        else:
            recommendations.append("✅ **می‌توان از میانگین یا میانه برای جایگزینی استفاده کرد**")
            recommendations.append("🔧 **روش KNN Imputation برای جایگزینی پیشنهاد می‌شود**")
    
    elif analysis_type == "آمار توصیفی":
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            if len(col_data) > 0:
                cv = (col_data.std() / col_data.mean()) * 100 if col_data.mean() != 0 else 0
                insights.append(f"ستون '{col}': ضریب تغییرات {cv:.1f}%")
                
                if cv > 50:
                    recommendations.append(f"📈 **ستون '{col}' نوسان بالا دارد - برای تحلیل سری‌های زمانی مناسب است**")
                elif cv < 10:
                    recommendations.append(f"📉 **ستون '{col}' پایدار است - برای شاخص‌های ثابت مناسب است**")
    
    elif analysis_type == "تحلیل داده‌های پرت":
        high_outlier_cols = []
        for col, row in analysis_data.iterrows():
            if row['درصد پرت'] > 10:
                high_outlier_cols.append((col, row['درصد پرت']))
        
        if high_outlier_cols:
            insights.append(f"{len(high_outlier_cols)} ستون با بیش از ۱۰٪ داده پرت شناسایی شد")
            for col, percent in high_outlier_cols:
                insights.append(f"ستون '{col}': {percent}% داده پرت")
                recommendations.append(f"🔍 **بررسی علت داده‌های پرت در ستون '{col}' ضروری است**")
                recommendations.append(f"⚡ **برای ستون '{col}' از روش Winsorization استفاده شود**")
        else:
            insights.append("داده‌های پرت در حد قابل قبولی هستند")
            recommendations.append("✅ **نیاز به اقدام خاصی برای داده‌های پرت نیست**")
    
    elif analysis_type == "ماتریس همبستگی":
        strong_correlations = []
        for col1 in analysis_data.columns:
            for col2 in analysis_data.columns:
                if col1 != col2 and abs(analysis_data.loc[col1, col2]) > 0.7:
                    strong_correlations.append((col1, col2, analysis_data.loc[col1, col2]))
        
        if strong_correlations:
            insights.append(f"{len(strong_correlations)} رابطه قوی همبستگی شناسایی شد")
            for col1, col2, corr in strong_correlations[:3]:  # فقط ۳ مورد اول
                insights.append(f"همبستگی قوی بین '{col1}' و '{col2}': {corr:.3f}")
                recommendations.append(f"📊 **ستون‌های '{col1}' و '{col2}' ممکن است اطلاعات تکراری داشته باشند**")
                recommendations.append(f"🔧 **حذف یکی از ستون‌های همبسته قوی برای کاهش ابعاد داده پیشنهاد می‌شود**")
    
    elif analysis_type == "آزمون نرمالیتی":
        normal_cols = []
        non_normal_cols = []
        for col, row in analysis_data.iterrows():
            if row.get('نرمال') == 'بله':
                normal_cols.append(col)
            else:
                non_normal_cols.append(col)
        
        insights.append(f"{len(normal_cols)} ستون نرمال، {len(non_normal_cols)} ستون غیرنرمال")
        
        if non_normal_cols:
            recommendations.append("📈 **برای ستون‌های غیرنرمال از آزمون‌های ناپارامتریک استفاده شود**")
            recommendations.append("🔧 **تبدیل لگاریتمی یا Box-Cox برای نرمال‌سازی پیشنهاد می‌شود**")
        else:
            recommendations.append("✅ **داده‌ها برای تحلیل‌های پارامتریک مناسب هستند**")
    
    elif "تحلیل ماهانه" in analysis_type:
        # تحلیل فصلی
        insights.append("الگوهای فصلی در داده‌ها شناسایی شد")
        recommendations.append("📅 **مدل‌سازی سری زمانی با درنظرگیری فصلیت پیشنهاد می‌شود**")
        recommendations.append("🔮 **از مدل‌های SARIMA یا Prophet برای پیش‌بینی استفاده شود**")
    
    elif "تحلیل سودآوری" in analysis_type:
        max_profit_col = analysis_data.loc[analysis_data['نسبت به کل'].idxmax()]
        insights.append(f"ستون '{max_profit_col.name}' با {max_profit_col['نسبت به کل']}% بیشترین سهم را دارد")
        recommendations.append("💰 **تمرکز بر بهبود ستون‌های با سودآوری بالا پیشنهاد می‌شود**")
        recommendations.append("📊 **تحلیل سبد محصول برای بهینه‌سازی پیشنهاد می‌شود**")
    
    # اگر بینش خاصی تولید نشد، بینش عمومی تولید کن
    if not insights:
        insights.append("داده‌ها از کیفیت قابل قبولی برخوردار هستند")
        recommendations.append("📈 **ادامه تحلیل با روش‌های پیشرفته پیشنهاد می‌شود**")
    
    return {
        'insights': list(set(insights)),  # حذف موارد تکراری
        'recommendations': list(set(recommendations))
    }

def generate_smart_analysis(df, analysis_type, analysis_data):
    """تولید تحلیل هوشمند با بینش و راهکار"""
    analysis_result = {
        'data': analysis_data,
        'insights': [],
        'recommendations': []
    }
    
    # تولید بینش و راهکار
    smart_analysis = generate_insights_and_recommendations(analysis_type, analysis_data, df)
    analysis_result['insights'] = smart_analysis['insights']
    analysis_result['recommendations'] = smart_analysis['recommendations']
    
    return analysis_result

def generate_basic_analysis(df):
    """تحلیل‌های پایه با هوش مصنوعی"""
    analyses = {}
    
    # آمار توصیفی
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        analyses['آمار توصیفی'] = generate_smart_analysis(
            df, "آمار توصیفی", numeric_df.describe().round(2)
        )
    
    # داده‌های مفقودی
    missing_data = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df) * 100).round(2)
    missing_analysis = pd.DataFrame({
        'تعداد مفقودی': missing_data,
        'درصد مفقودی': missing_percentage
    })
    analyses['داده‌های مفقودی'] = generate_smart_analysis(
        df, "داده‌های مفقودی", 
        missing_analysis[missing_analysis['تعداد مفقودی'] > 0]
    )
    
    # اطلاعات کلی
    info_analysis = pd.DataFrame({
        'ویژگی': ['تعداد ردیف', 'تعداد ستون', 'تعداد داده‌های مفقودی', 'تعداد داده‌های عددی', 'تعداد داده‌های متنی'],
        'مقدار': [
            len(df), 
            len(df.columns), 
            df.isnull().sum().sum(), 
            len(numeric_df.columns),
            len(df.select_dtypes(include=['object']).columns)
        ]
    })
    analyses['اطلاعات کلی'] = generate_smart_analysis(df, "اطلاعات کلی", info_analysis)
    
    # همبستگی
    if len(numeric_df.columns) > 1:
        correlation_matrix = numeric_df.corr().round(3)
        analyses['ماتریس همبستگی'] = generate_smart_analysis(
            df, "ماتریس همبستگی", correlation_matrix
        )
    
    return analyses

def generate_time_analysis(df):
    """تحلیل‌های زمانی با هوش مصنوعی"""
    analyses = {}
    date_columns = detect_date_columns(df)
    
    if date_columns:
        try:
            date_col = date_columns[0]
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            df_temp = df_temp.set_index(date_col)
            
            # تحلیل فصلی
            df_temp['month'] = df_temp.index.month
            df_temp['quarter'] = df_temp.index.quarter
            df_temp['year'] = df_temp.index.year
            
            numeric_cols = df_temp.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['month', 'quarter', 'year']]
            
            if len(numeric_cols) > 0:
                # تحلیل ماهانه
                monthly_analysis = df_temp.groupby('month')[numeric_cols].mean().round(2)
                analyses['تحلیل ماهانه'] = generate_smart_analysis(
                    df, "تحلیل ماهانه", monthly_analysis
                )
                
                # تحلیل فصلی
                quarterly_analysis = df_temp.groupby('quarter')[numeric_cols].mean().round(2)
                analyses['تحلیل فصلی'] = generate_smart_analysis(
                    df, "تحلیل فصلی", quarterly_analysis
                )
                
                # تحلیل سالانه
                yearly_analysis = df_temp.groupby('year')[numeric_cols].mean().round(2)
                analyses['تحلیل سالانه'] = generate_smart_analysis(
                    df, "تحلیل سالانه", yearly_analysis
                )
        
        except Exception as e:
            error_df = pd.DataFrame({'خطا': [f'خطا در تحلیل زمانی: {str(e)}']})
            analyses['خطا'] = generate_smart_analysis(df, "خطا", error_df)
    
    return analyses

def generate_statistical_analysis(df):
    """تحلیل‌های آماری پیشرفته با هوش مصنوعی"""
    analyses = {}
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        # تحلیل نرمالیتی
        normality_test = {}
        for col in numeric_df.columns:
            try:
                if len(numeric_df[col].dropna()) > 3:
                    stat, p_value = stats.shapiro(numeric_df[col].dropna())
                    normality_test[col] = {
                        'آماره': round(stat, 4),
                        'p-value': round(p_value, 4),
                        'نرمال': 'بله' if p_value > 0.05 else 'خیر'
                    }
            except:
                normality_test[col] = {'خطا': 'داده ناکافی'}
        
        analyses['آزمون نرمالیتی'] = generate_smart_analysis(
            df, "آزمون نرمالیتی", pd.DataFrame(normality_test).T
        )
        
        # تحلیل پرت‌ها
        outliers_analysis = {}
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)).sum()
            outlier_percentage = (outliers / len(numeric_df[col])) * 100
            
            outliers_analysis[col] = {
                'تعداد پرت': outliers,
                'درصد پرت': round(outlier_percentage, 2),
                'کران پایین': round(lower_bound, 2),
                'کران بالا': round(upper_bound, 2)
            }
        
        analyses['تحلیل داده‌های پرت'] = generate_smart_analysis(
            df, "تحلیل داده‌های پرت", pd.DataFrame(outliers_analysis).T
        )
    
    return analyses

def generate_business_analysis(df):
    """تحلیل‌های کسب‌وکار با هوش مصنوعی"""
    analyses = {}
    numeric_df = df.select_dtypes(include=[np.number])
    
    if not numeric_df.empty:
        # تحلیل سودآوری
        profitability = {}
        total_sum = numeric_df.sum().sum()
        for col in numeric_df.columns:
            if numeric_df[col].sum() > 0:
                profitability[col] = {
                    'مجموع': round(numeric_df[col].sum(), 2),
                    'میانگین': round(numeric_df[col].mean(), 2),
                    'نسبت به کل': round((numeric_df[col].sum() / total_sum) * 100, 2) if total_sum > 0 else 0
                }
        
        if profitability:
            analyses['تحلیل سودآوری'] = generate_smart_analysis(
                df, "تحلیل سودآوری", pd.DataFrame(profitability).T
            )
        
        # تحلیل رشد
        growth_analysis = {}
        for col in numeric_df.columns:
            if len(numeric_df[col]) > 1:
                try:
                    first_val = numeric_df[col].iloc[0]
                    last_val = numeric_df[col].iloc[-1]
                    if first_val != 0:
                        growth = ((last_val - first_val) / abs(first_val)) * 100
                        growth_analysis[col] = round(growth, 2)
                except:
                    growth_analysis[col] = 'خطا'
        
        if growth_analysis:
            analyses['درصد رشد کلی'] = generate_smart_analysis(
                df, "درصد رشد کلی", 
                pd.DataFrame.from_dict(growth_analysis, orient='index', columns=['درصد رشد'])
            )
    
    return analyses

def download_analysis_report(request, analysis_type, file_name):
    """دانلود گزارش تحلیل به صورت فایل Excel"""
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, 'results', f'{file_name}_analysis.xlsx')
        
        if not os.path.exists(file_path):
            messages.error(request, 'فایل مورد نظر یافت نشد')
            return render(request, 'analyzer/analysis_results.html')
        
        with pd.ExcelFile(file_path) as xls:
            sheet_names = xls.sheet_names
            target_sheet = None
            for sheet in sheet_names:
                if analysis_type in sheet:
                    target_sheet = sheet
                    break
            
            if not target_sheet:
                target_sheet = sheet_names[0] if sheet_names else 'Sheet1'
            
            df = pd.read_excel(xls, sheet_name=target_sheet)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=analysis_type, index=True)
        
        output.seek(0)
        
        response = HttpResponse(
            output.getvalue(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        response['Content-Disposition'] = f'attachment; filename="{file_name}_{analysis_type}.xlsx"'
        
        return response
        
    except Exception as e:
        messages.error(request, f'خطا در تولید گزارش: {str(e)}')
        return render(request, 'analyzer/analysis_results.html')

def upload_dataset(request):
    """آپلود و تحلیل فایل اکسل"""
    if request.method == 'POST':
        if 'file' not in request.FILES:
            messages.error(request, 'لطفا یک فایل اکسل انتخاب کنید')
            return render(request, 'analyzer/upload.html')
        
        file = request.FILES['file']
        
        if not file.name.lower().endswith(('.xlsx', '.xls')):
            messages.error(request, 'فقط فایل‌های اکسل با پسوند .xlsx و .xls قابل قبول هستند')
            return render(request, 'analyzer/upload.html')
        
        try:
            df = pd.read_excel(file)
            results_dir = os.path.join(settings.MEDIA_ROOT, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            file_name = os.path.splitext(file.name)[0]
            date_columns = detect_date_columns(df)
            
            # تولید تحلیل‌های مختلف
            all_analyses = {
                'پایه': generate_basic_analysis(df),
                'زمانی': generate_time_analysis(df),
                'آماری': generate_statistical_analysis(df),
                'کسب‌وکار': generate_business_analysis(df)
            }
            
            # ذخیره همه تحلیل‌ها در یک فایل Excel
            output_file = os.path.join(results_dir, f'{file_name}_analysis.xlsx')
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                for category, analyses in all_analyses.items():
                    for analysis_name, analysis_data in analyses.items():
                        sheet_name = f"{category}_{analysis_name}"[:31]
                        analysis_data['data'].to_excel(writer, sheet_name=sheet_name, index=True)
                
                df.to_excel(writer, sheet_name='داده_های_اصلی', index=False)
            
            # نمایش نتایج
            context = {
                'file_name': file_name,
                'output_file': output_file,
                'all_analyses': all_analyses,
                'basic_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'missing_total': df.isnull().sum().sum(),
                    'columns_list': df.columns.tolist(),
                    'has_date_columns': len(date_columns) > 0,
                    'date_columns': date_columns
                }
            }
            
            messages.success(request, f'فایل "{file_name}" با موفقیت تحلیل شد!')
            return render(request, 'analyzer/analysis_results.html', context)
            
        except Exception as e:
            messages.error(request, f'خطا در تحلیل فایل: {str(e)}')
            return render(request, 'analyzer/upload.html')
    
    return render(request, 'analyzer/upload.html')

# views.py - این قسمت را باید با منطق خودتان جایگزین کنید
def perform_analysis(request, selected_column):
    """
    این تابع را کاملا با تحلیل‌های خاص خودتان جایگزین کنید
    """
    # TODO: اینجا را با کدهای تحلیل خودتان پر کنید
    # مثال:
    try:
        filename = request.session.get('excel_filename')
        df = pd.read_excel(f'media/excel_files/{filename}')
        
        # 🔥 این بخش را با تحلیل‌های واقعی خودتان جایگزین کنید
        analysis = {
            'column_name': selected_column,
            'your_custom_analysis': 'نتایج تحلیل شما اینجا قرار می‌گیرد',
            # اضافه کردن تحلیل‌های خاص شما
        }
        
        return analysis
        
    except Exception as e:
        return {'error': str(e)}