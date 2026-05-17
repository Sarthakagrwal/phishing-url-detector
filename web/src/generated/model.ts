/**
 * AUTO-GENERATED — do not edit by hand.
 *
 * Source: models/model_meta.json
 * Generator: ml/export_js.py
 *
 * These are the parameters of the logistic-regression phishing classifier,
 * trained offline on the PhiUSIIL Phishing URL Dataset (UCI #967) by
 * ml/train.py. The browser runs the *identical* model as the Python CLI by
 * evaluating the closed-form sigmoid over these numbers in src/predict.ts:
 *
 *   z_i   = (x_i - mean_i) / scale_i
 *   logit = intercept + Σ_i coef_i · z_i
 *   p     = 1 / (1 + e^(-logit))
 *
 * Regenerate with:  python ml/export_js.py
 */

export interface PhishingModel {
  /** Ordered feature names; the coefficients are aligned to this order. */
  readonly featureNames: readonly string[];
  /** Per-feature logistic-regression weights. */
  readonly coef: readonly number[];
  /** Logistic-regression bias term. */
  readonly intercept: number;
  /** Per-feature mean from the fitted StandardScaler. */
  readonly mean: readonly number[];
  /** Per-feature standard deviation from the fitted StandardScaler. */
  readonly scale: readonly number[];
  /** Decision threshold on the phishing probability. */
  readonly threshold: number;
  /** Held-out test-set metrics recorded at training time. */
  readonly metrics: {
    readonly accuracy: number;
    readonly precision: number;
    readonly recall: number;
    readonly f1: number;
    readonly roc_auc: number;
  };
}

export const MODEL: PhishingModel = {
  featureNames: [
    "url_length", "hostname_length", "path_length", "num_dots",
    "num_hyphens", "num_subdomains", "has_ip_host", "has_at_symbol",
    "num_query_params", "has_punycode", "has_homograph", "is_https",
    "num_digits_in_host", "digit_ratio_host", "suspicious_tld", "is_shortener",
    "num_suspicious_keywords", "has_double_slash_in_path", "num_special_chars", "tld_length",
  ],
  coef: [
    3.152392121841019, 0.01225668843055501, -1.240149149530153, 0.8396871560776821, 0.7031894573348321, -0.33638522597427584, -0.038618338899019305, 0.6720376277105335, -0.7160182067646728, 0.21639504791107644, 0.0, -1.2428188709045236, 5.004364014558771, -1.158499085258049, 0.8857779796608821, 0.5581354456773491, 0.26809561250213315, 0.3820191474484523, 0.03861162409979746, -0.3388567334010248,
  ],
  intercept: 0.8050840930511342,
  mean: [
    36.636175710594316, 20.636375760606818, 6.055497207635242, 2.1468533800116694, 0.24252729849128948, 0.9876302408935568, 0.00208385429690756, 0.004801200300075019, 0.08910560973576727, 0.0003834291906309911, 0.0, 0.7362674001833792, 0.7119779944986246, 0.025494704998437347, 0.03359173126614987, 0.002350587646911728, 0.0367425189630741, 0.0014837042593981828, 0.21688755522213887, 3.010686004834542,
  ],
  scale: [
    41.39430239030117, 8.716788747565372, 32.376944535648505, 1.0292331966900348, 0.689309317016984, 0.5959606974741654, 0.04560166497156029, 0.06912415479231161, 0.8626885686565189, 0.019577593638810575, 1.0, 0.44065600598510074, 2.491083854778533, 0.08308829772924325, 0.1801758220646575, 0.048425844180811965, 0.2190030031352317, 0.0384902959337422, 3.085528318300292, 0.9094554864504711,
  ],
  threshold: 0.5,
  metrics: {
  accuracy: 0.8495699139827966,
  f1: 0.7931034482758621,
  precision: 0.7497832495231489,
  recall: 0.8417364220362079,
  roc_auc: 0.9073511248189072,
  },
};
