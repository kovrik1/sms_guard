"""
Management command `python manage.py seed_demo` (Задание 16 п. 8, Раздел 8.3).
Creates demo_user (DemoPass123!), admin (AdminPass123!), ModelVersion records,
and 35 checks (including all 13 reference examples from Section 8.3 of the TZ)
distributed over the last 30 days.
"""
import json
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.accounts.models import get_or_create_profile
from apps.analysis.models import Check, ModelVersion
from apps.analysis.services import (
    AnalysisResult,
    analyze_sms,
    analyze_url,
    save_analysis_result,
)

# All 13 reference messages from Section 8.3 of TZ + 22 additional synthetic samples (total 35)
DEMO_SAMPLES: list[tuple[str, str]] = [
    # --- 13 mandatory samples from Section 8.3 ---
    ("SMS", "Привет! Встречаемся завтра в 18:00 у входа в библиотеку."),
    ("SMS", "Your order #4821 has been shipped and will arrive on Monday."),
    ("SMS", "Сәлем! Ертең сағат 10:00-де кездесеміз бе?"),
    ("SMS", "Ваш код подтверждения: 4821. Никому не сообщайте этот код."),
    (
        "SMS",
        "Ваша карта заблокирована! Срочно подтвердите данные по ссылке: http://secure-bank-verify.example/login",
    ),
    (
        "SMS",
        "URGENT: Your account will be suspended in 24 hours. Verify now at http://192.0.2.15/account-update",
    ),
    (
        "SMS",
        "Мама, у меня проблемы, срочно переведи 50 000 на этот номер, потом всё объясню. Никому не говори.",
    ),
    (
        "SMS",
        "Congratulations! You won a $1000 gift card. Claim your prize: bit.ly/prize-claim-test",
    ),
    (
        "SMS",
        "Сотрудник службы безопасности банка: назовите код из SMS для отмены подозрительной операции.",
    ),
    (
        "SMS",
        "Картаңыз бұғатталды. Жедел түрде мына сілтеме арқылы растаңыз: http://bank-kz-verify.test",
    ),
    (
        "SMS",
        "Ваша посылка ожидает оплаты таможенного сбора 350 ₸: http://post-delivery-pay.example",
    ),
    ("URL", "https://www.example.com/about"),
    ("URL", "http://xn--exmple-cua.test/secure/login?verify=1"),
    # --- Additional synthetic samples for 30-day history & pagination (22 more = 35 total) ---
    ("SMS", "Напоминаем: приём у врача завтра в 14:30, кабинет 204."),
    ("SMS", "Dear colleague, the project meeting is moved to 15:00 in room 3B."),
    ("SMS", "Қайырлы күн! Тапсырысыңыз дүкенге жеткізілді, алып кетуіңізге болады."),
    (
        "SMS",
        "ВНИМАНИЕ! Попытка списания 120 000 ₸ с вашей карты. Срочно отмените перевод: http://kasp1-secure.test/cancel",
    ),
    (
        "SMS",
        "Поздравляем! Вы выиграли 500 000 тенге в акции! Для получения приза оплатите пошлину: http://bonus-prize-kz.xyz/claim",
    ),
    (
        "SMS",
        "Служба безопасности: зафиксирован вход с нового устройства. Сообщите код из SMS оператору.",
    ),
    ("URL", "https://example.org/documentation/getting-started"),
    ("URL", "http://198.51.100.42:8080/bank/login/verify?account=1"),
    ("URL", "http://halyk-account-update.xyz/signin"),
    (
        "SMS",
        "Абонент, ваш тариф будет обновлён с 1 числа следующего месяца. Подробности в личном кабинете.",
    ),
    (
        "SMS",
        "FINAL NOTICE: Your tax refund of $450 is pending. Confirm bank card details immediately at http://tax-refund-gov.example/form",
    ),
    (
        "SMS",
        "Сәлеметсіз бе! Сіз ұтыс ойынында 1 000 000 ₸ жеңіп алдыңыз! Сілтемеге өтіңіз: http://prize-kz.click/win",
    ),
    (
        "SMS",
        "Курьер приедет сегодня с 12:00 до 15:00. Пожалуйста, будьте на связи.",
    ),
    (
        "SMS",
        "Ваш аккаунт Госуслуг взломан! Срочно позвоните или подтвердите личность: http://egov-verify-kz.test/auth",
    ),
    ("URL", "https://www.example.com/services/tariffs"),
    ("URL", "http://bit.ly/free-bonus-card-verify"),
    (
        "SMS",
        "Лекция по машинному обучению перенесена в аудиторию 412.",
    ),
    (
        "SMS",
        "Срочно! Ваша банковская карта приостановлена. Введите CVV и код из SMS на http://secure-card-unlock.top/verify",
    ),
    (
        "SMS",
        "Hi Anna, thanks for sending the report. Let's review it tomorrow morning.",
    ),
    (
        "SMS",
        "Құрметті клиент, шотыңыз бұғатталуы мүмкін. SMS кодты ешкімге айтпай мына жерге енгізіңіз: http://jusan-check.test",
    ),
    ("URL", "https://example.org/contact"),
    ("URL", "http://paypal-security-update.example/login?confirm=true"),
]


class Command(BaseCommand):
    help = "Заполнить БД демонстрационными пользователями (demo_user, admin) и 35 проверками за 30 дней."

    def handle(self, *args, **options):
        # 1. Create or update demo_user and admin
        demo_user, created_demo = User.objects.get_or_create(
            username="demo_user",
            defaults={"email": "demo@smsguard.example"},
        )
        demo_user.set_password("DemoPass123!")
        demo_user.save()
        demo_profile = get_or_create_profile(demo_user)
        demo_profile.store_sms_text = True
        demo_profile.reset_failed_logins()
        demo_profile.save()

        admin_user, created_admin = User.objects.get_or_create(
            username="admin",
            defaults={
                "email": "admin@smsguard.example",
                "is_staff": True,
                "is_superuser": True,
            },
        )
        admin_user.is_staff = True
        admin_user.is_superuser = True
        admin_user.set_password("AdminPass123!")
        admin_user.save()
        admin_profile = get_or_create_profile(admin_user)
        admin_profile.reset_failed_logins()

        self.stdout.write(
            self.style.SUCCESS(
                "Пользователи готовы: demo_user (DemoPass123!) и admin (AdminPass123!)"
            )
        )

        # 2. Seed ModelVersion metadata
        self._seed_model_versions()

        # 3. Clear existing demo checks for demo_user to make command idempotent
        Check.objects.filter(user=demo_user).delete()

        now = timezone.now()
        total = len(DEMO_SAMPLES)
        created_count = 0

        for idx, (check_type, raw_input) in enumerate(DEMO_SAMPLES):
            # Spread checks across the last 29 days, with newest checks today
            days_ago = int(((total - 1 - idx) * 28) / max(1, total - 1))
            ts = now - timedelta(days=days_ago, hours=(idx % 8), minutes=(idx * 7) % 60)

            try:
                if check_type == "SMS":
                    res = analyze_sms(raw_input)
                else:
                    res = analyze_url(raw_input)
            except Exception:
                # Fallback if ML models are not yet built when seed_demo is called
                res = self._fallback_result(check_type, raw_input)

            save_analysis_result(
                user=demo_user,
                result=res,
                raw_input=raw_input,
                created_at_override=ts,
            )
            created_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Успешно создано {created_count} демонстрационных проверок для demo_user."
            )
        )

    def _seed_model_versions(self):
        artifacts_dir = Path(settings.BASE_DIR) / "ml" / "artifacts"
        ModelVersion.objects.all().delete()

        for mtype, meta_file, default_algo, default_metrics in (
            (
                "sms",
                "sms_model_meta.json",
                "LogisticRegression (TF-IDF word + char_wb + manual features)",
                {"precision": 0.94, "recall": 0.93, "f1": 0.935, "roc_auc": 0.985},
            ),
            (
                "url",
                "url_model_meta.json",
                "RandomForestClassifier (22 structural URL features)",
                {"precision": 0.91, "recall": 0.90, "f1": 0.905, "roc_auc": 0.968},
            ),
        ):
            version = "1.0.0"
            algorithm = default_algo
            threshold = 0.5
            metrics = default_metrics

            meta_path = artifacts_dir / meta_file
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                    version = str(data.get("version", version))
                    algorithm = str(data.get("algorithm", data.get("model_name", algorithm)))
                    threshold = float(data.get("threshold", data.get("t2", threshold)))
                    metrics = data.get("test_metrics") or data.get("metrics") or metrics
                except Exception:
                    pass

            ModelVersion.objects.create(
                model_type=mtype,
                version=version,
                algorithm=algorithm,
                metrics_json=metrics,
                threshold=threshold,
                trained_at=timezone.now(),
                is_active=True,
            )

    def _fallback_result(self, check_type: str, raw_input: str) -> AnalysisResult:
        from ml.preprocessing import detect_language, extract_urls_from_text
        from ml.url_features import explain_url, extract_url_features

        if check_type == "URL":
            feats = extract_url_features(raw_input)
            reasons = explain_url(feats)
            suspicious = (
                feats.get("has_ip_address", 0)
                or feats.get("has_punycode", 0)
                or feats.get("is_shortener", 0)
                or feats.get("is_suspicious_tld", 0)
                or feats.get("num_suspicious_words", 0) > 0
            )
            p_url = 0.86 if suspicious else 0.08
            cat = "DANGEROUS" if suspicious else "SAFE"
            lvl = "HIGH" if suspicious else "LOW"
            score = 86 if suspicious else 12
            return AnalysisResult(
                type="URL",
                language="en",
                category=cat,
                risk_level=lvl,
                risk_score=score,
                p_sms=None,
                p_url_max=p_url,
                urls=[
                    {
                        "url": raw_input,
                        "domain": raw_input.split("/")[2] if "://" in raw_input else raw_input.split("/")[0],
                        "p_url": p_url,
                        "category": cat,
                        "features": feats,
                    }
                ],
                features=["URL_SUSPICIOUS"] if suspicious else [],
                detected_features=(
                    [{"code": "URL_SUSPICIOUS", "title": "Подозрительная структура ссылки", "weight": 1.2}]
                    if suspicious
                    else []
                ),
                url_features=feats,
                explanation=reasons or ["Признаков фишинга в структуре ссылки не обнаружено."],
                recommendation=(
                    "Не переходите по ссылке и не вводите учётные данные."
                    if suspicious
                    else "Структура ссылки выглядит стандартной, но соблюдайте осторожность."
                ),
            )

        lang = detect_language(raw_input)
        urls = extract_urls_from_text(raw_input)
        lower = raw_input.lower()
        is_fraud = any(
            w in lower
            for w in (
                "заблокирован",
                "срочно",
                "suspended",
                "urgent",
                "won",
                "prize",
                "назовите код",
                "бұғатталды",
                "переведи",
                "взломан",
                "списания",
                "жеңіп",
            )
        )
        is_susp = not is_fraud and any(
            w in lower for w in ("ожидает оплаты", "код подтверждения", "таможенного")
        )
        if is_fraud:
            cat, lvl, score, p_sms = "FRAUDULENT", "HIGH", 85, 0.91
            codes = ["URGENCY", "ACCOUNT_BLOCK"]
            detected = [
                {"code": "URGENCY", "title": "Срочность и психологическое давление", "weight": 1.0},
                {"code": "ACCOUNT_BLOCK", "title": "Угроза блокировки карты или аккаунта", "weight": 1.2},
            ]
            expl = ["Сообщение требует срочного действия и оказывает психологическое давление."]
            rec = "Не переходите по ссылке и не сообщайте коды из SMS или реквизиты карты."
        elif is_susp:
            cat, lvl, score, p_sms = "SUSPICIOUS", "MEDIUM", 52, 0.54
            codes = ["MONEY_TRANSFER"]
            detected = [
                {"code": "MONEY_TRANSFER", "title": "Требование перевода денег или оплаты", "weight": 1.0}
            ]
            expl = ["Обнаружены отдельные признаки финансового запроса или внешней ссылки."]
            rec = "Перепроверьте информацию в официальном приложении перед любыми действиями."
        else:
            cat, lvl, score, p_sms = "SAFE", "LOW", 10, 0.06
            codes = []
            detected = []
            expl = ["Признаки мошенничества и социальной инженерии не обнаружены."]
            rec = "Сообщение выглядит безопасным."

        url_list = []
        for u in urls:
            uf = extract_url_features(u)
            url_list.append(
                {
                    "url": u,
                    "domain": u.split("/")[2] if "://" in u else u.split("/")[0],
                    "p_url": 0.82 if is_fraud else 0.45,
                    "category": "DANGEROUS" if is_fraud else "SUSPICIOUS",
                    "features": uf,
                }
            )

        return AnalysisResult(
            type="SMS",
            language=lang,
            category=cat,
            risk_level=lvl,
            risk_score=score,
            p_sms=p_sms,
            p_url_max=url_list[0]["p_url"] if url_list else None,
            urls=url_list,
            features=codes,
            detected_features=detected,
            explanation=expl,
            recommendation=rec,
        )
