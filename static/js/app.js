/**
 * SMS Guard Frontend Application Logic (ES6)
 * - Performs asynchronous SMS and URL checks via fetch() without page reload
 * - Strictly uses textContent and DOM creation (NEVER innerHTML) to prevent XSS (НФТ-5, Задание 15 п. 6)
 * - Renders non-clickable URLs with highlighted domain (ФТ-3.10)
 */

(function () {
    "use strict";

    const CATEGORY_META = {
        SAFE: {
            icon: "✓",
            label: "БЕЗОПАСНОЕ СООБЩЕНИЕ",
            urlLabel: "БЕЗОПАСНАЯ ССЫЛКА",
            badgeClass: "sg-badge-safe",
        },
        SUSPICIOUS: {
            icon: "⚠",
            label: "ПОДОЗРИТЕЛЬНОЕ СООБЩЕНИЕ",
            urlLabel: "ПОДОЗРИТЕЛЬНАЯ ССЫЛКА",
            badgeClass: "sg-badge-suspicious",
        },
        FRAUDULENT: {
            icon: "⛔",
            label: "МОШЕННИЧЕСКОЕ СООБЩЕНИЕ",
            urlLabel: "МОШЕННИЧЕСКАЯ ССЫЛКА",
            badgeClass: "sg-badge-fraudulent",
        },
        DANGEROUS: {
            icon: "⛔",
            label: "ПОТЕНЦИАЛЬНО ОПАСНАЯ ССЫЛКА",
            urlLabel: "ПОТЕНЦИАЛЬНО ОПАСНАЯ ССЫЛКА",
            badgeClass: "sg-badge-dangerous",
        },
    };

    const RISK_LEVEL_LABELS = {
        LOW: "Низкий",
        MEDIUM: "Средний",
        HIGH: "Высокий",
    };

    const LANGUAGE_LABELS = {
        ru: "Русский (ru)",
        kk: "Казахский (kk)",
        en: "Английский (en)",
        other: "Другой (other)",
    };

    const URL_FEATURE_SPEC = [
        { key: "url_length", num: 1, label: "Длина URL (символов)", type: "int", warnGt: 75 },
        { key: "domain_length", num: 2, label: "Длина домена (символов)", type: "int", warnGt: 30 },
        { key: "path_length", num: 3, label: "Длина пути URL", type: "int", warnGt: 40 },
        { key: "num_dots", num: 4, label: "Число точек в URL", type: "int", warnGt: 3 },
        { key: "num_subdomains", num: 5, label: "Число поддоменов (tldextract)", type: "int", warnGt: 2 },
        { key: "num_hyphens_domain", num: 6, label: "Число дефисов в домене", type: "int", warnGt: 1 },
        { key: "num_digits_url", num: 7, label: "Число цифр в URL", type: "int", warnGt: 6 },
        { key: "digit_ratio_domain", num: 8, label: "Доля цифр в домене", type: "float", warnGt: 0.15 },
        { key: "num_special_chars", num: 9, label: "Число спецсимволов (@ ~ % = & _ ?)", type: "int", warnGt: 3 },
        { key: "has_ip_address", num: 10, label: "IP-адрес вместо домена (IPv4/IPv6)", type: "flag", warnEq: 1 },
        { key: "is_https", num: 11, label: "Схема HTTPS", type: "https_flag", warnEq: 0 },
        { key: "has_non_standard_port", num: 12, label: "Нестандартный порт (не 80/443)", type: "flag", warnEq: 1 },
        { key: "has_at_symbol", num: 13, label: "Символ @ в URL", type: "flag", warnEq: 1 },
        { key: "has_double_slash_in_path", num: 14, label: "Двойной // в пути URL", type: "flag", warnEq: 1 },
        { key: "is_shortener", num: 15, label: "Сервис сокращения ссылок (bit.ly и др.)", type: "flag", warnEq: 1 },
        { key: "num_suspicious_words", num: 16, label: "Число подозрительных ключевых слов", type: "int", warnGt: 0 },
        { key: "has_punycode", num: 17, label: "Punycode (xn--) в домене", type: "flag", warnEq: 1 },
        { key: "has_mixed_scripts", num: 18, label: "Смешение алфавитов в домене", type: "flag", warnEq: 1 },
        { key: "brand_similarity", num: 19, label: "Сходство с брендом (Левенштейн ≤ 2)", type: "flag", warnEq: 1 },
        { key: "domain_entropy", num: 20, label: "Энтропия Шеннона символов домена", type: "float", warnGt: 3.8 },
        { key: "is_suspicious_tld", num: 21, label: "Подозрительная доменная зона (.xyz, .top, .test и др.)", type: "flag", warnEq: 1 },
        { key: "num_query_params", num: 22, label: "Число параметров запроса", type: "int", warnGt: 3 },
    ];

    function getCsrfToken() {
        const input = document.querySelector('input[name="csrfmiddlewaretoken"]');
        if (input && input.value) {
            return input.value;
        }
        const match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
        return match ? decodeURIComponent(match[1]) : "";
    }

    function clearChildren(el) {
        while (el && el.firstChild) {
            el.removeChild(el.firstChild);
        }
    }

    /**
     * Build a non-clickable span showing URL with visually highlighted domain (ФТ-3.10).
     */
    function createNonClickableUrlNode(rawUrl, domainHint) {
        const wrapper = document.createElement("span");
        wrapper.className = "url-nonclickable";

        const urlStr = String(rawUrl || "");
        let domain = String(domainHint || "").toLowerCase();
        if (!domain) {
            try {
                const withScheme = urlStr.includes("://") ? urlStr : "http://" + urlStr;
                const parsed = new URL(withScheme);
                domain = (parsed.hostname || "").toLowerCase();
            } catch (_e) {
                domain = "";
            }
        }

        const lowerUrl = urlStr.toLowerCase();
        const idx = domain ? lowerUrl.indexOf(domain) : -1;
        if (domain && idx !== -1) {
            const prefix = urlStr.slice(0, idx);
            const hostPart = urlStr.slice(idx, idx + domain.length);
            const suffix = urlStr.slice(idx + domain.length);

            if (prefix) {
                const preSpan = document.createElement("span");
                preSpan.textContent = prefix;
                wrapper.appendChild(preSpan);
            }
            const domStrong = document.createElement("strong");
            domStrong.className = "url-domain-highlight";
            domStrong.title = "Выделенный домен проверяемой ссылки";
            domStrong.textContent = hostPart;
            wrapper.appendChild(domStrong);

            if (suffix) {
                const sufSpan = document.createElement("span");
                sufSpan.textContent = suffix;
                wrapper.appendChild(sufSpan);
            }
        } else {
            wrapper.textContent = urlStr;
        }
        return wrapper;
    }

    /**
     * Populate the common analysis result panel using only textContent & createElement.
     */
    function renderAnalysisResult(data, isUrlCheck) {
        const resultContainer = document.getElementById("analysis-result-container");
        if (!resultContainer) return;
        resultContainer.classList.remove("d-none");

        const catKey = (data.category || "SAFE").toUpperCase();
        const catMeta = CATEGORY_META[catKey] || CATEGORY_META.SAFE;
        const riskLevel = (data.risk_level || "LOW").toUpperCase();
        const riskScore = Number(data.risk_score || 0);

        // 1. Category badge
        const badgeEl = document.getElementById("res-category-badge");
        if (badgeEl) {
            badgeEl.className = "sg-badge sg-badge-lg " + catMeta.badgeClass;
            badgeEl.textContent = catMeta.icon + " " + (isUrlCheck ? catMeta.urlLabel : catMeta.label);
        }

        // 2. Risk level & Risk Score
        const riskLevelEl = document.getElementById("res-risk-level");
        if (riskLevelEl) {
            riskLevelEl.className = "sg-badge sg-badge-" + riskLevel;
            riskLevelEl.textContent = "Уровень риска: " + (RISK_LEVEL_LABELS[riskLevel] || riskLevel);
        }

        const riskScoreValEl = document.getElementById("res-risk-score-val");
        if (riskScoreValEl) {
            riskScoreValEl.textContent = String(riskScore) + " / 100";
        }

        const riskBarEl = document.getElementById("res-risk-bar");
        if (riskBarEl) {
            riskBarEl.className = "sg-risk-meter-fill sg-risk-fill-" + riskLevel;
            riskBarEl.style.width = Math.max(2, Math.min(100, riskScore)) + "%";
        }

        // 3. Language & Language Warning
        const langEl = document.getElementById("res-language");
        if (langEl) {
            langEl.textContent = LANGUAGE_LABELS[data.language] || data.language || "—";
        }

        const langWarnBox = document.getElementById("res-language-warning");
        if (langWarnBox) {
            if (data.language_warning) {
                langWarnBox.textContent = "⚠ " + data.language_warning;
                langWarnBox.classList.remove("d-none");
            } else {
                langWarnBox.classList.add("d-none");
            }
        }

        // 4. Model Probabilities
        const pSmsEl = document.getElementById("res-p-sms");
        if (pSmsEl) {
            pSmsEl.textContent =
                data.p_sms !== null && data.p_sms !== undefined
                    ? (Number(data.p_sms) * 100).toFixed(1) + "% (" + Number(data.p_sms).toFixed(3) + ")"
                    : "Н/Д (проверка ссылки)";
        }

        const pUrlEl = document.getElementById("res-p-url");
        if (pUrlEl) {
            const pUrlVal = data.p_url_max !== null && data.p_url_max !== undefined ? data.p_url_max : data.p_url;
            pUrlEl.textContent =
                pUrlVal !== null && pUrlVal !== undefined
                    ? (Number(pUrlVal) * 100).toFixed(1) + "% (" + Number(pUrlVal).toFixed(3) + ")"
                    : "Ссылки не обнаружены";
        }

        // 5. Inspected input text / URL (strictly via textContent / non-clickable node!)
        const inputDisplayEl = document.getElementById("res-input-display");
        if (inputDisplayEl) {
            clearChildren(inputDisplayEl);
            const shownText = data.display_text || data.input_masked || "";
            if (isUrlCheck) {
                const firstDomain = data.urls && data.urls.length ? data.urls[0].domain : "";
                inputDisplayEl.appendChild(createNonClickableUrlNode(shownText, firstDomain));
            } else {
                inputDisplayEl.textContent = shownText;
            }
        }

        // 6. Detected Features
        const featuresListEl = document.getElementById("res-features-list");
        if (featuresListEl) {
            clearChildren(featuresListEl);
            const detFeatures = data.detected_features || [];
            if (detFeatures.length > 0) {
                detFeatures.forEach(function (df) {
                    const span = document.createElement("span");
                    span.className = "badge bg-secondary me-2 mb-2 py-2 px-3 text-wrap text-start";
                    span.textContent = df.code + ": " + df.title;
                    featuresListEl.appendChild(span);
                });
            } else if (Array.isArray(data.features) && data.features.length > 0) {
                data.features.forEach(function (code) {
                    const span = document.createElement("span");
                    span.className = "badge bg-secondary me-2 mb-2 py-2 px-3";
                    span.textContent = String(code);
                    featuresListEl.appendChild(span);
                });
            } else {
                const noneSpan = document.createElement("span");
                noneSpan.className = "text-muted";
                noneSpan.textContent = "Специфические маркеры социальной инженерии не сработали.";
                featuresListEl.appendChild(noneSpan);
            }
        }

        // 7. Explanation list
        const explListEl = document.getElementById("res-explanation-list");
        if (explListEl) {
            clearChildren(explListEl);
            const reasons = Array.isArray(data.explanation) ? data.explanation : [];
            reasons.forEach(function (reason) {
                const li = document.createElement("li");
                li.className = "list-group-item";
                li.textContent = String(reason);
                explListEl.appendChild(li);
            });
        }

        // 8. Extracted URLs block (for SMS check)
        const urlsSectionEl = document.getElementById("res-urls-section");
        const urlsListEl = document.getElementById("res-urls-list");
        if (urlsSectionEl && urlsListEl) {
            clearChildren(urlsListEl);
            const urls = Array.isArray(data.urls) ? data.urls : [];
            if (urls.length > 0) {
                urlsSectionEl.classList.remove("d-none");
                urls.forEach(function (uItem) {
                    const card = document.createElement("div");
                    card.className = "p-3 mb-2 border rounded bg-light d-flex flex-wrap justify-content-between align-items-center gap-2";

                    const leftDiv = document.createElement("div");
                    leftDiv.appendChild(createNonClickableUrlNode(uItem.url, uItem.domain));

                    const rightDiv = document.createElement("div");
                    rightDiv.className = "d-flex align-items-center gap-2";

                    const probSpan = document.createElement("span");
                    probSpan.className = "small fw-semibold text-secondary";
                    probSpan.textContent = "P_url = " + (Number(uItem.p_url || 0) * 100).toFixed(1) + "%";

                    const uCat = (uItem.category || "SAFE").toUpperCase();
                    const uMeta = CATEGORY_META[uCat] || CATEGORY_META.SAFE;
                    const uBadge = document.createElement("span");
                    uBadge.className = "sg-badge " + uMeta.badgeClass;
                    uBadge.textContent = uMeta.icon + " " + uMeta.urlLabel;

                    rightDiv.appendChild(probSpan);
                    rightDiv.appendChild(uBadge);

                    card.appendChild(leftDiv);
                    card.appendChild(rightDiv);
                    urlsListEl.appendChild(card);
                });
            } else {
                urlsSectionEl.classList.add("d-none");
            }
        }

        // 9. 22 URL features table (for standalone URL check)
        const urlFeaturesBody = document.getElementById("res-url-features-tbody");
        if (urlFeaturesBody && data.url_features) {
            clearChildren(urlFeaturesBody);
            const feats = data.url_features;
            URL_FEATURE_SPEC.forEach(function (spec) {
                const rawVal = feats[spec.key] !== undefined ? feats[spec.key] : 0;
                let triggered = false;
                if (spec.warnEq !== undefined) {
                    triggered = Number(Boolean(rawVal)) === spec.warnEq;
                } else if (spec.warnGt !== undefined) {
                    triggered = Number(rawVal) > spec.warnGt;
                }

                let displayVal = String(rawVal);
                if (spec.type === "flag") {
                    displayVal = rawVal ? "Да (1)" : "Нет (0)";
                } else if (spec.type === "https_flag") {
                    displayVal = rawVal ? "Да (HTTPS)" : "Нет (HTTP)";
                } else if (spec.type === "float") {
                    displayVal = Number(rawVal).toFixed(3);
                }

                const tr = document.createElement("tr");
                if (triggered) {
                    tr.className = "feature-row-triggered";
                }

                const tdNum = document.createElement("td");
                tdNum.textContent = String(spec.num);

                const tdName = document.createElement("td");
                tdName.textContent = spec.label;

                const tdVal = document.createElement("td");
                tdVal.className = "font-monospace";
                tdVal.textContent = displayVal;

                const tdStatus = document.createElement("td");
                const statusBadge = document.createElement("span");
                if (triggered) {
                    statusBadge.className = "sg-badge sg-badge-suspicious";
                    statusBadge.textContent = "⚠ Сработал";
                } else {
                    statusBadge.className = "badge bg-light text-secondary border";
                    statusBadge.textContent = "Норма";
                }
                tdStatus.appendChild(statusBadge);

                tr.appendChild(tdNum);
                tr.appendChild(tdName);
                tr.appendChild(tdVal);
                tr.appendChild(tdStatus);
                urlFeaturesBody.appendChild(tr);
            });
        }

        // 10. Recommendation & Disclaimer & Link to Detail
        const recEl = document.getElementById("res-recommendation");
        if (recEl) {
            recEl.textContent = data.recommendation || "";
        }

        const discEl = document.getElementById("res-disclaimer");
        if (discEl) {
            discEl.textContent =
                data.disclaimer || "Оценка носит вероятностный характер и не гарантирует безопасность";
        }

        const detailLinkEl = document.getElementById("res-detail-link");
        if (detailLinkEl && data.id) {
            detailLinkEl.href = "/checks/" + encodeURIComponent(String(data.id)) + "/";
            detailLinkEl.classList.remove("d-none");
        }
    }

    /**
     * Initialize SMS Check Page (/check/sms/)
     */
    function initSmsCheckPage() {
        const form = document.getElementById("sms-check-form");
        if (!form) return;

        const textarea = document.getElementById("sms-text-input");
        const charCounter = document.getElementById("sms-char-counter");
        const submitBtn = document.getElementById("sms-submit-btn");
        const clearBtn = document.getElementById("sms-clear-btn");
        const spinner = document.getElementById("sms-loading-spinner");
        const errorBox = document.getElementById("sms-error-box");
        const resultContainer = document.getElementById("analysis-result-container");

        function updateCounter() {
            const len = (textarea.value || "").length;
            charCounter.textContent = len + " / 1000";
            if (len > 1000) {
                charCounter.classList.add("text-danger", "fw-bold");
            } else {
                charCounter.classList.remove("text-danger", "fw-bold");
            }
        }

        textarea.addEventListener("input", updateCounter);
        updateCounter();

        // Example buttons
        document.querySelectorAll("[data-sms-example]").forEach(function (btn) {
            btn.addEventListener("click", function () {
                textarea.value = btn.getAttribute("data-sms-example") || "";
                updateCounter();
                errorBox.classList.add("d-none");
                textarea.focus();
            });
        });

        if (clearBtn) {
            clearBtn.addEventListener("click", function () {
                textarea.value = "";
                updateCounter();
                errorBox.classList.add("d-none");
                if (resultContainer) {
                    resultContainer.classList.add("d-none");
                }
                textarea.focus();
            });
        }

        form.addEventListener("submit", async function (e) {
            e.preventDefault();
            errorBox.classList.add("d-none");

            const rawText = textarea.value || "";
            if (!rawText.trim()) {
                errorBox.textContent = "Пожалуйста, введите текст SMS-сообщения (поле не может быть пустым).";
                errorBox.classList.remove("d-none");
                return;
            }
            if (rawText.length > 1000) {
                errorBox.textContent = "Длина текста SMS не должна превышать 1000 символов.";
                errorBox.classList.remove("d-none");
                return;
            }

            submitBtn.disabled = true;
            if (spinner) spinner.classList.remove("d-none");

            try {
                const response = await fetch("/api/v1/check/sms/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-CSRFToken": getCsrfToken(),
                    },
                    body: JSON.stringify({ text: rawText }),
                });

                const payload = await response.json();
                if (!response.ok) {
                    const msg =
                        (payload.error && payload.error.message) ||
                        "Ошибка при выполнении анализа SMS.";
                    errorBox.textContent = msg;
                    errorBox.classList.remove("d-none");
                    return;
                }

                renderAnalysisResult(payload, false);
            } catch (_err) {
                errorBox.textContent = "Ошибка соединения с сервером. Попробуйте повторить запрос.";
                errorBox.classList.remove("d-none");
            } finally {
                submitBtn.disabled = false;
                if (spinner) spinner.classList.add("d-none");
            }
        });
    }

    /**
     * Initialize URL Check Page (/check/url/)
     */
    function initUrlCheckPage() {
        const form = document.getElementById("url-check-form");
        if (!form) return;

        const urlInput = document.getElementById("url-text-input");
        const submitBtn = document.getElementById("url-submit-btn");
        const clearBtn = document.getElementById("url-clear-btn");
        const spinner = document.getElementById("url-loading-spinner");
        const errorBox = document.getElementById("url-error-box");
        const resultContainer = document.getElementById("analysis-result-container");

        document.querySelectorAll("[data-url-example]").forEach(function (btn) {
            btn.addEventListener("click", function () {
                urlInput.value = btn.getAttribute("data-url-example") || "";
                errorBox.classList.add("d-none");
                urlInput.focus();
            });
        });

        if (clearBtn) {
            clearBtn.addEventListener("click", function () {
                urlInput.value = "";
                errorBox.classList.add("d-none");
                if (resultContainer) {
                    resultContainer.classList.add("d-none");
                }
                urlInput.focus();
            });
        }

        form.addEventListener("submit", async function (e) {
            e.preventDefault();
            errorBox.classList.add("d-none");

            const rawUrl = (urlInput.value || "").trim();
            if (!rawUrl) {
                errorBox.textContent = "Пожалуйста, введите интернет-ссылку (URL) для проверки.";
                errorBox.classList.remove("d-none");
                return;
            }
            if (rawUrl.length > 2048) {
                errorBox.textContent = "Длина URL не должна превышать 2048 символов.";
                errorBox.classList.remove("d-none");
                return;
            }

            submitBtn.disabled = true;
            if (spinner) spinner.classList.remove("d-none");

            try {
                const response = await fetch("/api/v1/check/url/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-CSRFToken": getCsrfToken(),
                    },
                    body: JSON.stringify({ url: rawUrl }),
                });

                const payload = await response.json();
                if (!response.ok) {
                    const msg =
                        (payload.error && payload.error.message) ||
                        "Некорректный формат интернет-ссылки.";
                    errorBox.textContent = msg;
                    errorBox.classList.remove("d-none");
                    return;
                }

                renderAnalysisResult(payload, true);
            } catch (_err) {
                errorBox.textContent = "Ошибка соединения с сервером. Попробуйте повторить запрос.";
                errorBox.classList.remove("d-none");
            } finally {
                submitBtn.disabled = false;
                if (spinner) spinner.classList.add("d-none");
            }
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        initSmsCheckPage();
        initUrlCheckPage();
    });
})();
