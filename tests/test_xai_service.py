"""Tests for XAI (Explainable AI) services."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app.services.xai.shap_explainer import (
    prepare_features,
    train_surrogate_model,
    explain_predictions,
    analyze_feature_importance_by_group,
    _to_binary_series,
)
from app.services.xai.counterfactual import (
    generate_counterfactuals,
    find_minimum_changes,
    _get_prediction,
    _serialize_value,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe with predictions."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "feature3": np.random.randn(n),
        "gender": np.random.choice(["male", "female"], n),
        "age": np.random.randint(20, 60, n),
        "prediction": np.random.choice([0, 1], n),
    })


@pytest.fixture
def trained_model(sample_dataframe):
    """Create a trained model for testing."""
    X = sample_dataframe[["feature1", "feature2", "feature3", "age"]]
    y = sample_dataframe["prediction"]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


class TestPrepareFeatures:
    """Tests for feature preparation."""

    def test_prepare_features_numeric_only(self, sample_dataframe):
        """Test preparing numeric features."""
        features, encoders = prepare_features(
            sample_dataframe,
            ["feature1", "feature2", "age"],
            exclude_columns=["prediction"]
        )
        
        assert len(features) == len(sample_dataframe)
        assert list(features.columns) == ["feature1", "feature2", "age"]
        assert len(encoders) == 0  # No categorical columns
    
    def test_prepare_features_with_categorical(self, sample_dataframe):
        """Test preparing features with categorical columns."""
        features, encoders = prepare_features(
            sample_dataframe,
            ["feature1", "gender"],
            exclude_columns=["prediction"]
        )
        
        assert len(features) == len(sample_dataframe)
        assert "gender" in features.columns
        assert features["gender"].dtype in ["int64", "int32"]  # Encoded
        assert "gender" in encoders
    
    def test_prepare_features_missing_values(self, sample_dataframe):
        """Test handling missing values."""
        df = sample_dataframe.copy()
        df.loc[0, "feature1"] = np.nan
        
        features, _ = prepare_features(df, ["feature1"], exclude_columns=["prediction"])
        
        assert not features["feature1"].isna().any()  # Missing values filled


class TestTrainSurrogateModel:
    """Tests for surrogate model training."""

    def test_train_surrogate_random_forest(self, sample_dataframe):
        """Test training a random forest surrogate."""
        X, _ = prepare_features(
            sample_dataframe,
            ["feature1", "feature2", "feature3"],
            exclude_columns=["prediction"]
        )
        
        model = train_surrogate_model(X, sample_dataframe["prediction"])
        
        assert hasattr(model, "predict")
        assert model.n_classes_ == 2  # Binary classification
    
    def test_train_surrogate_continuous_targets(self, sample_dataframe):
        """Test training with continuous targets (auto-binarized)."""
        X, _ = prepare_features(
            sample_dataframe,
            ["feature1", "feature2"],
            exclude_columns=["prediction"]
        )
        
        # Create continuous targets
        y_continuous = sample_dataframe["feature1"]
        
        model = train_surrogate_model(X, y_continuous)
        
        assert hasattr(model, "predict")
        predictions = model.predict(X)
        assert set(np.unique(predictions)).issubset({0, 1})  # Binary output


class TestToBinarySeries:
    """Tests for binary conversion utility."""

    def test_binary_numeric(self):
        """Test converting numeric to binary."""
        s = pd.Series([0.5, 1.2, 0.3, 2.0])
        result = _to_binary_series(s)
        
        assert all(result.isin([0, 1]))
    
    def test_binary_boolean(self):
        """Test converting boolean to binary."""
        s = pd.Series([True, False, True])
        result = _to_binary_series(s)
        
        assert all(result.isin([0, 1]))
        assert result.iloc[0] == 1
        assert result.iloc[1] == 0
    
    def test_binary_string(self):
        """Test converting string labels to binary."""
        s = pd.Series(["yes", "no", "yes", "approved"])
        result = _to_binary_series(s)
        
        assert all(result.isin([0, 1]))


class TestExplainPredictions:
    """Tests for prediction explanation."""

    def test_explain_predictions_basic(self, sample_dataframe, trained_model):
        """Test basic prediction explanation."""
        result = explain_predictions(
            df=sample_dataframe,
            model=trained_model,
            prediction_col="prediction",
            feature_columns=["feature1", "feature2", "feature3", "age"],
            num_samples=3,
        )
        
        assert "explanations" in result
        assert len(result["explanations"]) == 3
        assert "feature_names" in result
        
        for exp in result["explanations"]:
            assert "index" in exp
            assert "prediction" in exp
            assert "top_positive_features" in exp
            assert "top_negative_features" in exp
            assert "explanation_summary" in exp
    
    def test_explain_predictions_with_indices(self, sample_dataframe, trained_model):
        """Test explaining specific indices."""
        result = explain_predictions(
            df=sample_dataframe,
            model=trained_model,
            prediction_col="prediction",
            feature_columns=["feature1", "feature2"],
            sample_indices=[0, 5, 10],
        )
        
        assert len(result["explanations"]) == 3
        indices = [e["index"] for e in result["explanations"]]
        assert indices == [0, 5, 10]


class TestAnalyzeFeatureImportanceByGroup:
    """Tests for per-group feature importance analysis."""

    def test_analyze_by_group(self, sample_dataframe, trained_model):
        """Test per-group feature importance."""
        result = analyze_feature_importance_by_group(
            df=sample_dataframe,
            model=trained_model,
            sensitive_column="gender",
            feature_columns=["feature1", "feature2", "feature3", "age"],
            exclude_columns=["prediction"],
        )
        
        assert "sensitive_attribute" in result
        assert result["sensitive_attribute"] == "gender"
        assert "groups" in result
        assert "male" in result["groups"] or "female" in result["groups"]
        assert "overall_top_features" in result
        assert "comparative_analysis" in result
    
    def test_analyze_bias_indicators(self, sample_dataframe, trained_model):
        """Test that bias indicators are detected."""
        result = analyze_feature_importance_by_group(
            df=sample_dataframe,
            model=trained_model,
            sensitive_column="gender",
            feature_columns=["feature1", "feature2", "feature3", "age"],
            exclude_columns=["prediction"],
        )
        
        assert "potential_bias_indicators" in result
        # Structure check
        if result["potential_bias_indicators"]:
            indicator = result["potential_bias_indicators"][0]
            assert "feature" in indicator
            assert "severity" in indicator
            assert "explanation" in indicator


class TestGenerateCounterfactuals:
    """Tests for counterfactual generation."""

    def test_generate_counterfactuals(self, sample_dataframe, trained_model):
        """Test basic counterfactual generation."""
        df = sample_dataframe.copy()
        df["prediction"] = trained_model.predict(
            df[["feature1", "feature2", "feature3", "age"]].fillna(0)
        )
        
        instance = df.iloc[0]
        current_pred = int(df.iloc[0]["prediction"])
        desired = 1 - current_pred
        
        result = generate_counterfactuals(
            df=df,
            model=trained_model,
            instance=instance,
            desired_outcome=desired,
            sensitive_columns=["gender"],
            num_counterfactuals=3,
        )
        
        assert "original_instance" in result
        assert "current_prediction" in result
        assert "desired_outcome" in result
        assert "counterfactuals" in result
        assert "bias_analysis" in result
        assert "summary" in result
    
    def test_counterfactual_bias_detection(self, sample_dataframe, trained_model):
        """Test that counterfactuals detect bias when sensitive attrs change."""
        df = sample_dataframe.copy()
        df["prediction"] = trained_model.predict(
            df[["feature1", "feature2", "feature3", "age"]].fillna(0)
        )
        
        instance = df.iloc[0]
        current_pred = int(df.iloc[0]["prediction"])
        desired = 1 - current_pred
        
        result = generate_counterfactuals(
            df=df,
            model=trained_model,
            instance=instance,
            desired_outcome=desired,
            sensitive_columns=["gender"],
            immutable_features=["gender"],
            num_counterfactuals=5,
        )
        
        assert "bias_analysis" in result
        bias = result["bias_analysis"]
        
        if bias["would_need_to_change_sensitive"]:
            assert len(bias["sensitive_attributes_in_changes"]) > 0
            assert "fairness_concerns" in bias
            assert len(bias["fairness_concerns"]) > 0


class TestFindMinimumChanges:
    """Tests for minimum change calculation."""

    def test_find_minimum_changes(self, sample_dataframe, trained_model):
        """Test finding minimum feature changes."""
        df = sample_dataframe.copy()
        
        instance = df.iloc[0]
        current_pred = int(_get_prediction(trained_model, instance[["feature1", "feature2", "feature3", "age"]].fillna(0)))
        desired = 1 - current_pred
        
        # Define feature ranges
        feature_ranges = {
            "feature1": (float(df["feature1"].min()), float(df["feature1"].max())),
            "feature2": (float(df["feature2"].min()), float(df["feature2"].max())),
            "feature3": (float(df["feature3"].min()), float(df["feature3"].max())),
            "age": (20.0, 60.0),
        }
        
        result = find_minimum_changes(
            df=df,
            model=trained_model,
            instance=instance,
            desired_outcome=desired,
            feature_ranges=feature_ranges,
        )
        
        assert "original_prediction" in result
        assert "desired_outcome" in result
        assert "minimal_changes" in result
        assert "total_features_changed" in result
        assert "success" in result
        assert "explanation" in result


class TestUtilities:
    """Tests for utility functions."""

    def test_serialize_value_numeric(self):
        """Test serializing numeric values."""
        assert _serialize_value(42) == 42.0
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value(np.int64(5)) == 5.0
        assert _serialize_value(np.float64(2.5)) == 2.5
    
    def test_serialize_value_special(self):
        """Test serializing special values."""
        assert _serialize_value(None) is None
        assert _serialize_value("test") == "test"
        assert _serialize_value(np.nan) is None
    
    def test_get_prediction(self, sample_dataframe, trained_model):
        """Test getting prediction from model."""
        instance = sample_dataframe.iloc[0]
        pred = _get_prediction(
            trained_model,
            instance[["feature1", "feature2", "feature3", "age"]].fillna(0)
        )
        
        assert pred in [0, 1]


class TestAPIIntegration:
    """Integration tests for XAI API endpoints."""
    
    def test_explain_info_endpoint(self, client):
        """Test the explain info endpoint returns correct structure."""
        response = client.get("/api/explain/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "endpoints" in data
        assert "supported_file_formats" in data
        assert "note" in data
        
        # Check endpoint structure
        endpoints = data["endpoints"]
        assert len(endpoints) >= 4
        
        for endpoint in endpoints:
            assert "path" in endpoint
            assert "method" in endpoint
            assert "description" in endpoint
