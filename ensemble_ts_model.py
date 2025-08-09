import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
from typing import List, Dict, Tuple
import joblib
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("🚀 Enhanced Multi-GPU/CPU Ensemble Stock Prediction System")
print("=" * 60)


class TechnicalIndicators:
    """Calculate comprehensive technical indicators"""

    @staticmethod
    def sma(data: np.array, window: int) -> np.array:
        """Simple Moving Average"""
        return pd.Series(data).rolling(window=window, min_periods=1).mean().values

    @staticmethod
    def ema(data: np.array, window: int) -> np.array:
        """Exponential Moving Average"""
        return pd.Series(data).ewm(span=window, min_periods=1).mean().values

    @staticmethod
    def rsi(data: np.array, window: int = 14) -> np.array:
        """Relative Strength Index"""
        delta = np.diff(data)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return np.concatenate([[50], rsi.values])  # First value as neutral

    @staticmethod
    def bollinger_bands(data: np.array, window: int = 20) -> Tuple[np.array, np.array, np.array]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = pd.Series(data).rolling(window=window, min_periods=1).std().values
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        return upper, sma, lower

    @staticmethod
    def macd(data: np.array, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.array, np.array]:
        """MACD"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        return macd_line, signal_line

    @staticmethod
    def volatility(data: np.array, window: int = 20) -> np.array:
        """Rolling volatility"""
        returns = np.diff(data) / data[:-1]
        vol = pd.Series(np.concatenate([[0], returns])).rolling(window=window, min_periods=1).std().values
        return vol * np.sqrt(252)  # Annualized


class FeatureEngineering:
    """Advanced feature engineering for stock data"""

    def __init__(self):
        self.scalers = {}

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        df_features = df.copy()

        # Price-based features
        df_features['returns'] = df_features['Close'].pct_change().fillna(0)
        df_features['log_returns'] = np.log(df_features['Close'] / df_features['Close'].shift(1)).fillna(0)
        df_features['high_low_ratio'] = df_features['High'] / df_features['Low']
        df_features['open_close_ratio'] = df_features['Open'] / df_features['Close']

        # Volume features
        df_features['volume_ma'] = TechnicalIndicators.sma(df_features['Volume'].values, 20)
        df_features['volume_ratio'] = df_features['Volume'] / (df_features['volume_ma'] + 1e-8)

        # Technical indicators
        df_features['sma_5'] = TechnicalIndicators.sma(df_features['Close'].values, 5)
        df_features['sma_10'] = TechnicalIndicators.sma(df_features['Close'].values, 10)
        df_features['sma_20'] = TechnicalIndicators.sma(df_features['Close'].values, 20)
        df_features['sma_50'] = TechnicalIndicators.sma(df_features['Close'].values, 50)

        df_features['ema_12'] = TechnicalIndicators.ema(df_features['Close'].values, 12)
        df_features['ema_26'] = TechnicalIndicators.ema(df_features['Close'].values, 26)

        df_features['rsi'] = TechnicalIndicators.rsi(df_features['Close'].values)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df_features['Close'].values)
        df_features['bb_upper'] = bb_upper
        df_features['bb_middle'] = bb_middle
        df_features['bb_lower'] = bb_lower
        df_features['bb_width'] = (bb_upper - bb_lower) / (bb_middle + 1e-8)
        df_features['bb_position'] = (df_features['Close'] - bb_lower) / ((bb_upper - bb_lower) + 1e-8)

        # MACD
        macd, macd_signal = TechnicalIndicators.macd(df_features['Close'].values)
        df_features['macd'] = macd
        df_features['macd_signal'] = macd_signal
        df_features['macd_histogram'] = macd - macd_signal

        # Volatility
        df_features['volatility'] = TechnicalIndicators.volatility(df_features['Close'].values)

        # Lag features (reduced to prevent overfitting)
        for lag in [1, 2, 3, 5]:
            df_features[f'close_lag_{lag}'] = df_features['Close'].shift(lag)
            df_features[f'returns_lag_{lag}'] = df_features['returns'].shift(lag)

        # Time-based features
        df_features['day_of_week'] = df_features['Date'].dt.dayofweek
        df_features['month'] = df_features['Date'].dt.month
        df_features['quarter'] = df_features['Date'].dt.quarter
        df_features['is_month_end'] = df_features['Date'].dt.is_month_end.astype(int)

        # Fill remaining NaN values
        df_features = df_features.fillna(method='bfill').fillna(method='ffill')

        return df_features


class LSTMAttentionModel(nn.Module):
    """Advanced LSTM with Attention Mechanism"""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_size * 2, num_heads=8, dropout=dropout, batch_first=True
        )

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Use the last timestep
        output = self.fc(attn_out[:, -1, :])
        return output


class TransformerModel(nn.Module):
    """Transformer Encoder for Time Series"""

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        seq_len = x.size(1)

        # Project to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)

        # Transformer
        x = self.transformer(x)

        # Output
        output = self.fc(x[:, -1, :])
        return output


class CNNLSTMModel(nn.Module):
    """CNN-LSTM Hybrid Model"""

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()

        # CNN layers
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # LSTM
        self.lstm = nn.LSTM(128, hidden_size, batch_first=True, dropout=dropout)

        # Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        # Back to (batch, seq_len, features)
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Output
        output = self.fc(lstm_out[:, -1, :])
        return output


class StockDataset(Dataset):
    """Custom dataset for stock data"""

    def __init__(self, features: np.array, targets: np.array, sequence_length: int = 60):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class EnsembleMetaLearner(nn.Module):
    """Meta-learner that combines predictions from multiple models"""

    def __init__(self, num_models: int, hidden_size: int = 64):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_models + 10, hidden_size),  # +10 for additional features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 2)  # mean and std predictions
        )

    def forward(self, model_predictions, additional_features):
        x = torch.cat([model_predictions, additional_features], dim=1)
        output = self.fc(x)
        mean_pred = output[:, 0:1]
        std_pred = torch.exp(output[:, 1:2])  # Ensure positive std
        return mean_pred, std_pred


class EnhancedEnsemblePredictor:
    """Enhanced ensemble predictor with better GPU/CPU utilization"""

    def __init__(self, data_path: str, sequence_length: int = 60):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.feature_engineer = FeatureEngineering()
        self.models = {}
        self.scalers = {}

        # Enhanced device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🔥 Using GPU: {torch.cuda.get_device_name()}")
            print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = torch.device("cpu")
            print(f"💻 Using CPU with {mp.cpu_count()} cores")

        # Set CPU threads for traditional models
        self.n_jobs = mp.cpu_count()

    def load_and_prepare_data(self):
        """Load and prepare data with features"""
        # Load data
        df = pd.read_csv(self.data_path)
        df.rename(columns={'Close/Last': 'Close'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Clean price columns
        price_columns = ["Open", "High", "Low", "Close"]
        for col in price_columns:
            df[col] = df[col].replace(r'[\$,]', '', regex=True).astype(float)

        df['Volume'] = df['Volume'].replace(r'[,]', '', regex=True).astype(float)

        print(f"📊 Dataset loaded: {len(df)} rows from {df['Date'].min()} to {df['Date'].max()}")

        # Feature engineering
        print("🔧 Creating features...")
        df_features = self.feature_engineer.create_features(df)

        # Select features for modeling (remove problematic features)
        excluded_cols = ['Date', 'Close']
        feature_cols = [col for col in df_features.columns if col not in excluded_cols]

        # Remove features with infinite or very large values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(method='ffill').fillna(method='bfill')

        # Scale features with robust scaling
        scaler_features = StandardScaler()
        scaler_target = MinMaxScaler()

        features_scaled = scaler_features.fit_transform(df_features[feature_cols].values)
        target_scaled = scaler_target.fit_transform(df_features[['Close']].values).flatten()

        self.scalers['features'] = scaler_features
        self.scalers['target'] = scaler_target
        self.feature_cols = feature_cols
        self.df = df_features

        print(f"✅ Features created: {len(feature_cols)} features")

        return features_scaled, target_scaled

    def create_datasets(self, features, targets, train_split=0.8):
        """Create train/validation datasets"""
        # Create sequences
        dataset = StockDataset(features, targets, self.sequence_length)

        # Split data
        train_size = int(len(dataset) * train_split)
        train_data = torch.utils.data.Subset(dataset, range(train_size))
        val_data = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        print(f"📈 Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

        return train_loader, val_loader, dataset

    def train_neural_models(self, train_loader, val_loader, num_features):
        """Train all neural network models with enhanced GPU utilization"""
        models_config = {
            'lstm_attention': LSTMAttentionModel(num_features),
            'transformer': TransformerModel(num_features),
            'cnn_lstm': CNNLSTMModel(num_features)
        }

        for name, model in models_config.items():
            print(f"\n🤖 Training {name} on GPU...")
            model = model.to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)
            criterion = nn.MSELoss()

            best_val_loss = float('inf')
            patience = 0
            max_patience = 10

            for epoch in range(50):
                # Training
                model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device, non_blocking=True), batch_y.to(self.device,
                                                                                              non_blocking=True)

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_loss += loss.item()

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device, non_blocking=True), batch_y.to(self.device,
                                                                                                  non_blocking=True)
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience = 0
                    torch.save(model.state_dict(), f'best_{name}.pth')
                else:
                    patience += 1

                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}, Val Loss: {avg_val_loss:.6f}")

                if patience >= max_patience:
                    break

            # Load best model
            model.load_state_dict(torch.load(f'best_{name}.pth'))
            self.models[name] = model
            print(f"  ✅ {name} training completed. Best Val Loss: {best_val_loss:.6f}")

    def train_traditional_models(self, features, targets):
        """Train traditional ML models with enhanced CPU utilization"""
        # Prepare data for traditional models
        X, y = [], []
        for i in range(self.sequence_length, len(features)):
            # Use aggregated features instead of flattened sequence to reduce dimensionality
            sequence = features[i - self.sequence_length:i]

            # Create aggregated features: mean, std, min, max, first, last
            agg_features = []
            for feat_idx in range(sequence.shape[1]):
                feat_values = sequence[:, feat_idx]
                agg_features.extend([
                    np.mean(feat_values),
                    np.std(feat_values),
                    np.min(feat_values),
                    np.max(feat_values),
                    feat_values[0],  # First value
                    feat_values[-1]  # Last value
                ])

            X.append(agg_features)
            y.append(targets[i])

        X, y = np.array(X), np.array(y)

        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        print(f"📊 Traditional ML input shape: {X_train.shape}")

        # Random Forest with CPU parallelization
        print(f"\n🌲 Training Random Forest with {self.n_jobs} CPU cores...")
        rf_model = RandomForestRegressor(
            n_estimators=100,  # Reduced for stability
            max_depth=8,  # Reduced to prevent overfitting
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=self.n_jobs
        )
        rf_model.fit(X_train, y_train)
        rf_score = rf_model.score(X_val, y_val)
        self.models['random_forest'] = rf_model
        print(f"  ✅ Random Forest R² Score: {rf_score:.4f}")

        # XGBoost with GPU support if available
        print(f"\n⚡ Training XGBoost...")
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': self.n_jobs
        }

        # Enable GPU for XGBoost if available
        if torch.cuda.is_available():
            try:
                xgb_params['tree_method'] = 'gpu_hist'
                xgb_params['gpu_id'] = 0
                print("  🔥 Using XGBoost GPU acceleration")
            except:
                print("  💻 XGBoost falling back to CPU")

        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train)
        xgb_score = xgb_model.score(X_val, y_val)
        self.models['xgboost'] = xgb_model
        print(f"  ✅ XGBoost R² Score: {xgb_score:.4f}")

        return X, y

    def train_meta_learner(self, val_loader, X_val, y_val):
        """Fixed meta-learner training"""
        print("\n🧠 Training Meta-Learner...")

        # Collect predictions from neural models
        all_neural_predictions = []
        all_additional_features = []
        all_targets = []

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_predictions = []

            # Get neural model predictions
            for name in ['lstm_attention', 'transformer', 'cnn_lstm']:
                if name in self.models:
                    model = self.models[name]
                    model.eval()
                    with torch.no_grad():
                        pred = model(batch_x).cpu().numpy()
                        batch_predictions.append(pred)

            if batch_predictions:
                neural_preds = np.concatenate(batch_predictions, axis=1)
                all_neural_predictions.append(neural_preds)

                # Additional features
                last_prices = batch_x[:, -5:, 3].cpu().numpy()  # Last 5 close prices
                volatility = np.std(last_prices, axis=1, keepdims=True)
                trend = (last_prices[:, -1:] - last_prices[:, 0:1]) / (last_prices[:, 0:1] + 1e-8)
                momentum = np.mean(np.diff(last_prices, axis=1), axis=1, keepdims=True)

                additional_feats = np.concatenate([
                    volatility, trend, momentum,
                    np.mean(last_prices, axis=1, keepdims=True),
                    np.std(last_prices, axis=1, keepdims=True),
                    np.min(last_prices, axis=1, keepdims=True),
                    np.max(last_prices, axis=1, keepdims=True),
                    last_prices[:, -1:],
                    last_prices[:, -2:-1],
                    (last_prices[:, -1:] - last_prices[:, -2:-1])
                ], axis=1)

                all_additional_features.append(additional_feats)
                all_targets.append(batch_y.numpy())

        # Convert to arrays
        neural_preds = np.vstack(all_neural_predictions)
        additional_feats = np.vstack(all_additional_features)
        targets = np.vstack(all_targets)

        # Get traditional model predictions on matching validation data
        val_size = len(neural_preds)
        X_val_subset = X_val[-val_size:] if len(X_val) >= val_size else X_val

        # Ensure we have predictions for the same samples
        min_size = min(len(neural_preds), len(X_val_subset))

        if min_size > 0:
            rf_preds = self.models['random_forest'].predict(X_val_subset[:min_size]).reshape(-1, 1)
            xgb_preds = self.models['xgboost'].predict(X_val_subset[:min_size]).reshape(-1, 1)

            # Combine all predictions
            ensemble_input = np.concatenate([
                neural_preds[:min_size],
                rf_preds,
                xgb_preds
            ], axis=1)

            # Create meta-learner dataset
            meta_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(ensemble_input),
                torch.FloatTensor(additional_feats[:min_size]),
                torch.FloatTensor(targets[:min_size])
            )
            meta_loader = DataLoader(meta_dataset, batch_size=32, shuffle=True)

            # Train meta-learner
            meta_learner = EnsembleMetaLearner(ensemble_input.shape[1]).to(self.device)
            optimizer = torch.optim.Adam(meta_learner.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            for epoch in range(100):
                meta_learner.train()
                total_loss = 0

                for model_preds, add_feats, targets_batch in meta_loader:
                    model_preds = model_preds.to(self.device)
                    add_feats = add_feats.to(self.device)
                    targets_batch = targets_batch.to(self.device)

                    optimizer.zero_grad()
                    mean_pred, std_pred = meta_learner(model_preds, add_feats)

                    mse_loss = criterion(mean_pred, targets_batch)
                    uncertainty_loss = torch.mean(std_pred)
                    loss = mse_loss + 0.1 * uncertainty_loss

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                if epoch % 20 == 0:
                    print(f"  Meta-learner Epoch {epoch}, Loss: {total_loss / len(meta_loader):.6f}")

            self.models['meta_learner'] = meta_learner
            print("  ✅ Meta-learner training completed!")
        else:
            print("  ❌ Insufficient data for meta-learner training")

    def predict_future(self, num_days: int = 30) -> Tuple[np.array, np.array, np.array]:
        """Generate future predictions with uncertainty"""
        print(f"\n🔮 Generating {num_days}-day predictions...")

        # Get last sequence
        features_scaled = self.scalers['features'].transform(self.df[self.feature_cols].values)
        last_sequence = features_scaled[-self.sequence_length:]

        predictions = []
        uncertainties = []

        current_sequence = last_sequence.copy()

        for day in range(num_days):
            # Neural model predictions
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
            neural_preds = []

            for name in ['lstm_attention', 'transformer', 'cnn_lstm']:
                if name in self.models:
                    model = self.models[name]
                    model.eval()
                    with torch.no_grad():
                        pred = model(input_tensor).cpu().numpy()
                        neural_preds.append(pred)

            # Traditional model predictions
            agg_features = []
            for feat_idx in range(current_sequence.shape[1]):
                feat_values = current_sequence[:, feat_idx]
                agg_features.extend([
                    np.mean(feat_values),
                    np.std(feat_values),
                    np.min(feat_values),
                    np.max(feat_values),
                    feat_values[0],
                    feat_values[-1]
                ])

            flat_sequence = np.array(agg_features).reshape(1, -1)
            rf_pred = self.models['random_forest'].predict(flat_sequence)[0]
            xgb_pred = self.models['xgboost'].predict(flat_sequence)[0]

            # Combine predictions
            all_preds = np.concatenate([
                np.concatenate(neural_preds, axis=1)[0],
                [rf_pred, xgb_pred]
            ])

            # Additional features for meta-learner
            last_prices = current_sequence[-5:, 3]
            volatility = np.std(last_prices)
            trend = (last_prices[-1] - last_prices[0]) / (last_prices[0] + 1e-8)
            momentum = np.mean(np.diff(last_prices))

            additional_feats = np.array([
                volatility, trend, momentum,
                np.mean(last_prices), np.std(last_prices),
                np.min(last_prices), np.max(last_prices),
                last_prices[-1], last_prices[-2],
                last_prices[-1] - last_prices[-2]
            ]).reshape(1, -1)

            # Meta-learner prediction
            if 'meta_learner' in self.models:
                meta_learner = self.models['meta_learner']
                meta_learner.eval()
                with torch.no_grad():
                    ensemble_input = torch.FloatTensor(all_preds).unsqueeze(0).to(self.device)
                    add_feats = torch.FloatTensor(additional_feats).to(self.device)
                    mean_pred, std_pred = meta_learner(ensemble_input, add_feats)

                    final_pred = mean_pred.cpu().numpy()[0, 0]
                    uncertainty = std_pred.cpu().numpy()[0, 0]
            else:
                # Fallback: simple ensemble average
                final_pred = np.mean(all_preds)
                uncertainty = np.std(all_preds)

            # Add realistic noise
            noise = np.random.normal(0, uncertainty * 0.3)
            final_pred_with_noise = final_pred + noise

            predictions.append(final_pred_with_noise)
            uncertainties.append(uncertainty)

            # Update sequence
            new_row = current_sequence[-1].copy()
            # Find close price index in features
            close_idx = None
            for i, col in enumerate(self.feature_cols):
                if 'Close' in col or col == 'Close':
                    close_idx = i
                    break

            if close_idx is not None:
                new_row[close_idx] = final_pred_with_noise

            current_sequence = np.vstack([current_sequence[1:], new_row])

        # Convert back to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_original = self.scalers['target'].inverse_transform(predictions).flatten()
        uncertainties = np.array(uncertainties)

        # Create confidence intervals
        scale_factor = self.scalers['target'].scale_[0] * 100
        confidence_upper = predictions_original + (uncertainties * scale_factor)
        confidence_lower = predictions_original - (uncertainties * scale_factor)

        return predictions_original, confidence_upper, confidence_lower

    def visualize_predictions(self, predictions: np.array, upper_bound: np.array,
                              lower_bound: np.array, target_date: str = None):
        """Create comprehensive visualization"""
        # Create future dates
        last_date = self.df['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                     periods=len(predictions), freq='D')

        # Setup the plot
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('🚀 Enhanced Multi-GPU/CPU Ensemble Stock Prediction', fontsize=16, fontweight='bold')

        # Plot 1: Full historical + predictions
        ax1 = axes[0, 0]
        historical_data = self.df.tail(500)
        ax1.plot(historical_data['Date'], historical_data['Close'],
                 label='Historical Price', color='blue', linewidth=1, alpha=0.8)
        ax1.plot(future_dates, predictions,
                 label='AI Ensemble Forecast', color='red', linewidth=2.5)
        ax1.fill_between(future_dates, lower_bound, upper_bound,
                         alpha=0.3, color='red', label='Confidence Interval')

        if target_date:
            target_dt = pd.to_datetime(target_date)
            if target_dt in future_dates:
                ax1.axvline(x=target_dt, color='green', linestyle='--',
                            linewidth=2, alpha=0.8, label='Target Date')

        ax1.set_title('Historical Performance + AI Forecast')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Recent focus
        ax2 = axes[0, 1]
        recent_data = self.df.tail(100)
        ax2.plot(recent_data['Date'], recent_data['Close'],
                 label='Recent Historical', color='blue', linewidth=1.5)
        ax2.plot(future_dates, predictions,
                 label='AI Forecast', color='red', linewidth=2.5)
        ax2.fill_between(future_dates, lower_bound, upper_bound,
                         alpha=0.3, color='red')

        if target_date:
            target_dt = pd.to_datetime(target_date)
            if target_dt in future_dates:
                ax2.axvline(x=target_dt, color='green', linestyle='--', linewidth=2)

        ax2.set_title('Recent Performance + Forecast (Zoomed)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Prediction uncertainty
        ax3 = axes[1, 0]
        uncertainty_pct = ((upper_bound - lower_bound) / predictions) * 100
        ax3.plot(future_dates, uncertainty_pct, color='purple', linewidth=2, marker='o', markersize=4)
        ax3.set_title('Prediction Uncertainty (%)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Uncertainty (%)')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Price distribution
        ax4 = axes[1, 1]
        ax4.hist(predictions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(np.mean(predictions), color='red', linestyle='--',
                    label=f'Mean: ${np.mean(predictions):.2f}')
        ax4.axvline(np.median(predictions), color='orange', linestyle='--',
                    label=f'Median: ${np.median(predictions):.2f}')
        ax4.set_title('Predicted Price Distribution')
        ax4.set_xlabel('Price ($)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def train_complete_ensemble(self):
        """Train the complete enhanced ensemble system"""
        print("🚀 Starting Enhanced Multi-GPU/CPU Ensemble Training...")

        # Load and prepare data
        features, targets = self.load_and_prepare_data()

        # Create datasets
        train_loader, val_loader, dataset = self.create_datasets(features, targets)

        # Train neural models on GPU
        num_features = features.shape[1]
        self.train_neural_models(train_loader, val_loader, num_features)

        # Train traditional models on CPU
        X, y = self.train_traditional_models(features, targets)

        # Train meta-learner
        val_size = len(val_loader.dataset)
        X_val = X[-val_size:] if len(X) >= val_size else X
        y_val = y[-val_size:] if len(y) >= val_size else y
        self.train_meta_learner(val_loader, X_val, y_val)

        # Save the complete ensemble
        self.save_ensemble()

        print("\n🎉 Enhanced ensemble training completed successfully!")

    def save_ensemble(self):
        """Save the trained ensemble"""
        ensemble_data = {
            'scalers': self.scalers,
            'feature_cols': self.feature_cols,
            'sequence_length': self.sequence_length
        }
        joblib.dump(ensemble_data, 'enhanced_ensemble_metadata.pkl')
        print("💾 Enhanced ensemble saved successfully!")


# === MAIN EXECUTION ===
def main():
    # Initialize the enhanced ensemble predictor
    data_path = r"D:\1 STUDY DRIVE\6 CODINGS\Other Projects\stock_predict_AI\data\alphabet.csv"
    ensemble = EnhancedEnsemblePredictor(data_path, sequence_length=60)

    # Train the complete ensemble
    ensemble.train_complete_ensemble()

    # Generate predictions
    predictions, upper_bound, lower_bound = ensemble.predict_future(num_days=30)

    # User input for target date
    print("\n" + "=" * 60)
    last_available_date = ensemble.df['Date'].max().date()
    print(f"📅 Last available date in dataset: {last_available_date}")

    target_date_str = input("Enter a future date to predict stock price (YYYY-MM-DD): ")
    target_date = datetime.datetime.strptime(target_date_str, '%Y-%m-%d').date()

    if target_date <= last_available_date:
        print("❌ Target date must be after the last date in the dataset.")
        return

    date_delta = (target_date - last_available_date).days - 1
    if date_delta < 0 or date_delta >= len(predictions):
        print(f"❌ Prediction range is up to {len(predictions)} days after the last available date.")
        return

    predicted_price = predictions[date_delta]
    upper_confidence = upper_bound[date_delta]
    lower_confidence = lower_bound[date_delta]

    print(f"\n🎯 **ENHANCED ENSEMBLE PREDICTION RESULTS**")
    print(f"🔥 GPU Acceleration: {'✅ ENABLED' if torch.cuda.is_available() else '❌ DISABLED'}")
    print(f"💻 CPU Cores Used: {mp.cpu_count()}")
    print(f"📊 Predicted price on {target_date}: ${predicted_price:.2f}")
    print(f"📈 Upper confidence bound: ${upper_confidence:.2f}")
    print(f"📉 Lower confidence bound: ${lower_confidence:.2f}")
    print(f"🎲 Prediction range: ${lower_confidence:.2f} - ${upper_confidence:.2f}")
    print(f"📏 Confidence interval: ±${(upper_confidence - lower_confidence) / 2:.2f}")

    # Visualize results
    ensemble.visualize_predictions(predictions, upper_bound, lower_bound, target_date_str)

    # Model performance summary
    print(f"\n📋 **ENHANCED ENSEMBLE SUMMARY**")
    print(f"   🤖 Neural Models: LSTM+Attention, Transformer, CNN-LSTM (GPU)")
    print(f"   🌲 Traditional Models: Random Forest, XGBoost (Multi-CPU + GPU)")
    print(f"   🧠 Meta-Learning: Neural ensemble with uncertainty (GPU)")
    print(f"   📊 Features: {len(ensemble.feature_cols)} engineered indicators")
    print(f"   📈 Training Data: {len(ensemble.df)} days of historical data")
    print(f"   ⏱️ Sequence Length: {ensemble.sequence_length} days")
    print(f"   🔮 Prediction Horizon: {len(predictions)} days")
    print(f"   📏 Average Uncertainty: ±{np.mean((upper_bound - lower_bound) / 2):.2f}")


if __name__ == "__main__":
    main()
