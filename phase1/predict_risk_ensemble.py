"""
Ensemble Risk Classification Prediction System
Uses LightGBM, XGBoost, and CatBoost models with majority voting

Name: Risk Prediction System
Date: 2025
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from pathlib import Path
from typing import Dict, Union, List, Optional
from collections import Counter
from datetime import datetime
import warnings

# Suppress specific pandas warnings regarding downcasting
pd.set_option('future.no_silent_downcasting', True)

# Import preprocessor helper functions and class from the base file
# We'll copy them here for standalone usage


def normalize_ws(s):
    """Normalize whitespace in strings."""
    if pd.isna(s):
        return s
    return " ".join(str(s).split())


def parse_possible_datetimes(df):
    """Lightweight parse for obvious date strings."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            s = df[c].astype("string")
            if s.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", regex=True, na=False).mean() > 0.2:
                try:
                    df[c] = pd.to_datetime(
                        s, errors="coerce", infer_datetime_format=True)
                except Exception:
                    pass
    return df


def expand_datetimes(df):
    """Expand datetime columns into year, month, day, etc."""
    df = df.copy()
    dt_cols = [
        c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for c in dt_cols:
        s = df[c]
        df[f"{c}__year"] = s.dt.year.astype("Int16")
        df[f"{c}__month"] = s.dt.month.astype("Int8")
        df[f"{c}__day"] = s.dt.day.astype("Int8")
        df[f"{c}__dow"] = s.dt.dayofweek.astype("Int8")
        df[f"{c}__hour"] = s.dt.hour.fillna(0).astype("Int8")
        df[f"{c}__mstart"] = s.dt.is_month_start.astype("Int8")
        df[f"{c}__mend"] = s.dt.is_month_end.astype("Int8")
    df.drop(columns=dt_cols, inplace=True)
    return df


def downcast_numeric(df):
    """Downcast numeric columns to save memory."""
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
    return df


def pd_cat_fix(series, allowed):
    """Coerce series to categorical with fixed categories; unseen -> 'Missing'."""
    s = series.astype("string").fillna("Missing")
    s = s.where(s.isin(allowed), "Missing")
    return pd.Categorical(s, categories=allowed, ordered=False)


class TabularPreprocessor:
    """
    Fits on train, then applies consistent transforms to val/test/inference:
    - normalize whitespace
    - downcast numeric
    - parse + expand datetimes, drop originals
    - convert non-numeric to categorical with a 'Missing' bucket
    - store feature order, cat columns & categories, numeric columns
    """

    def __init__(self):
        self.feature_names_ = None
        self.cat_cols_ = []
        self.cat_categories_ = {}
        self.num_cols_ = []
        self.fitted_ = False

    def _prep_base(self, df):
        d = df.copy()
        for c in d.columns:
            if d[c].dtype == "object":
                d[c] = d[c].map(normalize_ws)
        d = parse_possible_datetimes(d)
        d = expand_datetimes(d)
        for c in d.columns:
            if pd.api.types.is_bool_dtype(d[c]):
                d[c] = d[c].astype("int8")
        d = downcast_numeric(d)
        return d

    def fit(self, X: pd.DataFrame):
        d = self._prep_base(X)
        self.num_cols_ = [
            c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
        self.cat_cols_ = [c for c in d.columns if c not in self.num_cols_]

        for c in self.cat_cols_:
            s = d[c].astype("string").fillna("Missing")
            categories = pd.Index(pd.unique(s.dropna()))
            if "Missing" not in categories:
                categories = categories.insert(len(categories), "Missing")
            self.cat_categories_[c] = categories.tolist()
            d[c] = pd_cat_fix(s, self.cat_categories_[c])

        self.feature_names_ = self.num_cols_ + self.cat_cols_
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame):
        assert self.fitted_, "Preprocessor not fitted."
        d = self._prep_base(X)
        for c in self.feature_names_:
            if c not in d.columns:
                d[c] = np.nan
        d = d[self.feature_names_].copy()
        for c in self.num_cols_:
            d[c] = pd.to_numeric(d[c], errors="coerce")
            if d[c].isna().any():
                d[c] = d[c].fillna(d[c].median())
        for c in self.cat_cols_:
            allowed = self.cat_categories_[c]
            d[c] = pd_cat_fix(d[c], allowed)
        return d


class EnsembleRiskPredictor:
    """
    Ensemble predictor using LightGBM, XGBoost, and CatBoost models
    Uses majority voting to determine final prediction
    """

    def __init__(self,
                 lightgbm_dir: str = "./LightGBM",
                 xgboost_dir: str = "./XgBoost",
                 catboost_dir: str = "./CatBoost"):
        """
        Initialize ensemble predictor

        Args:
            lightgbm_dir: Path to folder containing LightGBM model
            xgboost_dir: Path to folder containing XGBoost model
            catboost_dir: Path to folder containing CatBoost model
        """
        self.lightgbm_dir = Path(lightgbm_dir)
        self.xgboost_dir = Path(xgboost_dir)
        self.catboost_dir = Path(catboost_dir)

        # Models
        self.lightgbm_model = None
        self.xgboost_model = None
        self.catboost_model = None

        # Preprocessors (we'll use LightGBM's preprocessor as reference)
        self.preprocessor = None

        # Metadata
        self.metadata = None
        self.num_classes = None

        # Load all models and files
        self._load_all_models()

    def _load_all_models(self):
        """Load all three models and their preprocessors"""
        print("=" * 60)
        print("Loading Ensemble Models...")
        print("=" * 60)

        # Load LightGBM
        self._load_lightgbm()

        # Load XGBoost
        self._load_xgboost()

        # Load CatBoost
        self._load_catboost()

        # Load metadata (use LightGBM's metadata as reference)
        self._load_metadata()

        print("=" * 60)
        print("✅ All models loaded successfully!")
        print("=" * 60)
        print()

    def _load_lightgbm(self):
        """Load LightGBM model"""
        model_path = self.lightgbm_dir / "lightgbm_booster.pkl"
        preprocessor_path = self.lightgbm_dir / "preprocessor.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"LightGBM model not found at path: {model_path}"
            )
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"LightGBM preprocessor not found at path: {preprocessor_path}"
            )

        # Load model
        with open(model_path, 'rb') as f:
            self.lightgbm_model = pickle.load(f)

        # Load preprocessor (we'll use this as the main preprocessor)
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)

        print(f"✅ LightGBM model loaded from: {model_path}")

    def _load_xgboost(self):
        """Load XGBoost model"""
        model_path = self.xgboost_dir / "xgboost_model-001.json"
        preprocessor_path = self.xgboost_dir / "preprocessor.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at path: {model_path}"
            )
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"XGBoost preprocessor not found at path: {preprocessor_path}"
            )

        # Load XGBoost model from JSON
        self.xgboost_model = xgb.Booster()
        self.xgboost_model.load_model(str(model_path))

        print(f"✅ XGBoost model loaded from: {model_path}")

    def _load_catboost(self):
        """Load CatBoost model"""
        model_path = self.catboost_dir / "catboost_multiclass.cbm"
        preprocessor_path = self.catboost_dir / "preprocessor.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"CatBoost model not found at path: {model_path}"
            )
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"CatBoost preprocessor not found at path: {preprocessor_path}"
            )

        # Load CatBoost model from CBM file
        self.catboost_model = cb.CatBoostClassifier()
        self.catboost_model.load_model(str(model_path))

        print(f"✅ CatBoost model loaded from: {model_path}")

    def _load_metadata(self):
        """Load metadata"""
        metadata_path = self.lightgbm_dir / "training_meta.json"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at path: {metadata_path}"
            )

        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)

        self.num_classes = self.metadata.get('num_classes', 10)
        print(f"✅ Metadata loaded. Number of risk classes: {self.num_classes}")

    def _fill_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill NULL values with default values

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with default values filled
        """
        df = df.copy()

        # Numeric features: NULL -> 0 or np.nan
        numeric_features = [
            'col', 'col_1', 'col_3', 'col_6', 'col_10', 'col_11', 'col_12', 'col_13',
            'col_14', 'col_15', 'col_17', 'col_18', 'col_19', 'col_20', 'col_21',
            'col_25', 'col_27', 'col_29', 'col_36', 'col_37', 'col_38', 'col_40',
            'col_41', 'col_42', 'col_43', 'col_45', 'col_46', 'col_47', 'col_48',
            'col_49', 'col_50', 'col_51'
        ]

        # Categorical features: NULL -> 'Missing'
        categorical_features = [
            'col_2', 'col_4', 'col_5', 'col_7', 'col_8', 'col_9', 'col_16',
            'col_22', 'col_23', 'col_24', 'col_26', 'col_28', 'col_30', 'col_31',
            'col_32', 'col_33', 'col_34', 'col_35', 'col_39', 'col_44'
        ]

        # Fill NULL values for numeric features
        for col in numeric_features:
            if col in df.columns:
                # Use infer_objects(copy=False) to avoid FutureWarning about silent downcasting
                df[col] = df[col].fillna(0).infer_objects(copy=False)
                df[col] = df[col].replace([None, np.nan], 0)

        # Fill NULL values for categorical features
        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('Missing').infer_objects(copy=False)
                df[col] = df[col].replace([None, np.nan], 'Missing')

        return df

    def _prepare_input(self, input_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare input data

        Args:
            input_data: Input data as Dictionary or DataFrame

        Returns:
            DataFrame ready for preprocessing
        """
        # If it's a Dictionary, convert to DataFrame
        if isinstance(input_data, dict):
            # If it's a single sample (single row)
            if all(not isinstance(v, (list, np.ndarray)) for v in input_data.values()):
                df = pd.DataFrame([input_data])
            else:
                df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(
                "Input must be Dictionary or DataFrame!\n"
                f"Input type: {type(input_data)}"
            )

        # Fill NULL values with default values
        df = self._fill_defaults(df)

        return df

    def predict(self, input_data: Union[Dict, pd.DataFrame],
                return_proba: bool = False) -> Union[int, np.ndarray, Dict]:
        """
        Predict risk class using ensemble (majority voting)

        Args:
            input_data: Input data (Dictionary or DataFrame)
            return_proba: If True, also returns probability for each class

        Returns:
            - If return_proba=False: Predicted class (int or array)
            - If return_proba=True: dictionary containing class, probabilities, and individual model predictions
        """
        # Prepare data
        df = self._prepare_input(input_data)

        # Preprocessing
        try:
            X_processed = self.preprocessor.transform(df)
        except Exception as e:
            raise ValueError(
                f"Error in data preprocessing:\n{str(e)}\n"
                "Please make sure all required features are present."
            )

        # Get predictions from all three models
        lightgbm_proba = self.lightgbm_model.predict(X_processed)

        # --- FIX: Enable Categorical support for XGBoost DMatrix ---
        # XGBoost requires 'enable_categorical=True' to handle Pandas Category dtypes
        xgboost_proba = self.xgboost_model.predict(
            xgb.DMatrix(X_processed, enable_categorical=True)
        )

        catboost_proba = self.catboost_model.predict_proba(X_processed)

        # Get predicted classes from each model
        lightgbm_class = int(np.argmax(lightgbm_proba[0]))
        xgboost_class = int(np.argmax(xgboost_proba[0]))
        catboost_class = int(np.argmax(catboost_proba[0]))

        # Majority voting
        votes = [lightgbm_class, xgboost_class, catboost_class]
        vote_counts = Counter(votes)
        ensemble_class = vote_counts.most_common(1)[0][0]

        # Calculate confidence (agreement percentage)
        agreement = vote_counts[ensemble_class] / len(votes)

        # Average probabilities for ensemble
        ensemble_proba = (
            lightgbm_proba[0] + xgboost_proba[0] + catboost_proba[0]) / 3.0
        ensemble_confidence = float(np.max(ensemble_proba))

        if return_proba:
            return {
                'risk_class': ensemble_class,
                'confidence': ensemble_confidence,
                'agreement': agreement,  # Percentage of models that agreed
                'individual_predictions': {
                    'lightgbm': lightgbm_class,
                    'xgboost': xgboost_class,
                    'catboost': catboost_class
                },
                'individual_probabilities': {
                    'lightgbm': {f'class_{i}': float(prob) for i, prob in enumerate(lightgbm_proba[0])},
                    'xgboost': {f'class_{i}': float(prob) for i, prob in enumerate(xgboost_proba[0])},
                    'catboost': {f'class_{i}': float(prob) for i, prob in enumerate(catboost_proba[0])}
                },
                'probabilities': {
                    f'class_{i}': float(prob) for i, prob in enumerate(ensemble_proba)
                }
            }
        else:
            return ensemble_class


def _sort_column_key(col_name: str) -> tuple:
    """
    Helper function to sort column names properly.
    Returns a tuple for sorting: (is_col, number)
    - 'col' comes first
    - 'col_X' sorted by X numerically
    """
    if col_name == 'col':
        return (0, 0)
    elif col_name.startswith('col_'):
        try:
            num = int(col_name.split('_', 1)[1])
            return (1, num)
        except ValueError:
            return (2, 0)
    else:
        return (3, 0)


def _read_json_input(input_path: Optional[str] = None) -> Dict:
    """
    Read JSON input from file or stdin

    Args:
        input_path: Path to JSON file. If None, reads from stdin

    Returns:
        Dictionary with input values
    """
    if input_path:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)

    # Extract values from JSON structure
    if 'values' in data:
        return data['values']
    else:
        return data


def main():
    """
    Main function for testing and usage
    Reads JSON input and outputs JSON with all columns + risk_class_score
    """
    print("=" * 60)
    print("🚀 Ensemble Risk Classification Prediction System")
    print("   Using LightGBM + XGBoost + CatBoost")
    print("=" * 60)
    print()

    # Determine input source
    input_file = None
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        print("Usage: python predict_risk_ensemble.py <input_json_file>")
        sys.exit(1)

    # Read JSON input
    try:
        input_data = _read_json_input(input_file)
        print(f"✅ Input JSON loaded from: {input_file}")
        print()
    except Exception as e:
        print(f"❌ Error reading input JSON: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Create ensemble predictor
    try:
        predictor = EnsembleRiskPredictor(
            lightgbm_dir="./LightGBM",
            xgboost_dir="./XgBoost",
            catboost_dir="./CatBoost"
        )
        print()
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Prediction
    try:
        print("🔮 Predicting with ensemble...")
        result = predictor.predict(input_data, return_proba=True)
        risk_class_score = result['risk_class']

        print()
        print("=" * 60)
        print("📊 Ensemble Prediction Result:")
        print("=" * 60)

        # Display input data (only features with real values)
        filtered_data = {k: v for k, v in input_data.items() if v is not None}
        if filtered_data:
            print(f"your data was: {filtered_data}")
        else:
            print(f"your data was: {input_data}")
        print()

        # Display ensemble risk class
        print(f"risk_score_class: {risk_class_score}")
        print(
            f"ensemble_agreement: {result['agreement']:.2%} ({result['agreement']*100:.1f}% of models agreed)")
        print()

        # Display individual model predictions
        print("Individual model predictions:")
        for model_name, pred_class in result['individual_predictions'].items():
            print(f"  {model_name}: class {pred_class}")
        print()

        # Display probabilities for each model
        print("=" * 60)
        print("📈 Individual Model Probabilities:")
        print("=" * 60)
        for model_name in ['lightgbm', 'xgboost', 'catboost']:
            print(f"\n{model_name.upper()} Probabilities:")
            model_probs = result['individual_probabilities'][model_name]
            # Sort by class number for better readability
            sorted_probs = sorted(model_probs.items(), key=lambda x: int(
                x[0].split('_')[1]) if '_' in x[0] else 0)
            for class_name, prob in sorted_probs:
                class_num = class_name.split(
                    '_')[1] if '_' in class_name else '0'
                print(f"  Class {class_num}: {prob:.6f} ({prob*100:.4f}%)")
        print()

        # Display ensemble probabilities
        print("=" * 60)
        print("📊 Ensemble Average Probabilities:")
        print("=" * 60)
        sorted_ensemble_probs = sorted(result['probabilities'].items(
        ), key=lambda x: int(x[0].split('_')[1]) if '_' in x[0] else 0)
        for class_idx, prob in sorted_ensemble_probs:
            class_num = class_idx.split('_')[1] if '_' in class_idx else '0'
            print(f"  Class {class_num}: {prob:.6f} ({prob*100:.4f}%)")

        print()
        print("=" * 60)

        # Sort columns in order: col, col_1, col_2, col_3, ...
        sorted_columns = sorted(input_data.keys(), key=_sort_column_key)

        # Build output JSON with all input columns + risk_class_score
        output_values = {}
        for col in sorted_columns:
            output_values[col] = input_data[col]
        output_values['risk_class_score'] = risk_class_score

        # Create final output structure
        output = {
            "values": output_values
        }

        # Generate output filename with date
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"ml_result-{current_date}.json"

        # Save JSON to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"✅ Result saved to: {output_filename}")
        print("=" * 60)
        print("✅ Ensemble prediction completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
