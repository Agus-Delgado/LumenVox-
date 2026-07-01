# LumenVox — Data Dictionary

The raw dataset (`data/raw/customer_feedback_raw.csv`) contains **13 columns**. The processed dataset (`data/processed/customer_feedback_clean.csv`) contains **23 columns** — the original 13 plus 10 derived analytical fields.

| column_name | type | source | description | example |
|-------------|------|--------|-------------|---------|
| `feedback_id` | string | raw + processed | Unique identifier for each feedback record | `FB-00001` |
| `date` | date (YYYY-MM-DD) | raw + processed | Date the feedback was submitted | `2025-12-13` |
| `channel` | string (categorical) | raw + processed | Channel through which feedback was received | `contact_form` |
| `customer_segment` | string (categorical) | raw + processed | Customer segment classification | `small_business` |
| `plan_type` | string (categorical) | raw + processed | Subscription plan tier | `free` |
| `language` | string (categorical) | raw + processed | Language code of the message | `es` |
| `message` | string | raw + processed | Original open-text customer feedback | `El precio es justo por el valor que obtenemos de la plataforma. Estamos en el plan gratuito pero esperamos un mínimo de estabilidad.` |
| `rating` | integer (1–5) | raw + processed | Customer satisfaction rating | `4` |
| `product_area` | string (categorical) | raw + processed | Product area the feedback relates to | `pricing` |
| `sentiment_label` | string (categorical) | raw + processed | Sentiment classification: positive, neutral, or negative | `positive` |
| `priority` | string (categorical) | raw + processed | Support priority level: low, medium, high, or critical | `low` |
| `resolved` | boolean | raw + processed | Whether the feedback issue has been resolved | `True` |
| `response_time_hours` | float | raw + processed | Hours elapsed before a response was provided | `15.2` |
| `clean_message` | string | processed | Normalized message text (lowercased, URLs removed, punctuation stripped) | `el precio es justo por el valor que obtenemos de la plataforma estamos en el plan gratuito pero esperamos un mínimo de estabilidad` |
| `message_length` | integer | processed | Character count of `clean_message` | `130` |
| `word_count` | integer | processed | Word count of `clean_message` (0 for empty messages) | `23` |
| `month` | integer (1–12) | processed | Month extracted from `date` | `12` |
| `is_negative` | boolean | processed | True when `sentiment_label == "negative"` | `False` |
| `is_positive` | boolean | processed | True when `sentiment_label == "positive"` | `True` |
| `is_critical` | boolean | processed | True when `priority == "critical"` | `False` |
| `is_high_priority` | boolean | processed | True when `priority == "high"` | `False` |
| `is_unresolved` | boolean | processed | True when `resolved == False` | `False` |
| `is_unresolved_critical` | boolean | processed | True when feedback is both critical and unresolved | `False` |
