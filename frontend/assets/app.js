(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.get("api")) {
        try {
            window.localStorage.setItem("speechshield_api_url", params.get("api"));
        } catch {}
    }
    const storedApi = (() => {
        try {
            return window.localStorage.getItem("speechshield_api_url");
        } catch {
            return null;
        }
    })();
    const API_BASE = (
        params.get("api") ||
        storedApi ||
        (window.location.protocol === "file:"
            ? "http://localhost:8000"
            : window.location.port === "8501"
              ? `${window.location.protocol}//${window.location.hostname}:8000`
              : window.location.origin)
    ).replace(/\/+$/, "");

    const readStorage = (key, fallback) => {
        try {
            return window.localStorage.getItem(key) ?? fallback;
        } catch {
            return fallback;
        }
    };

    const writeStorage = (key, value) => {
        try {
            window.localStorage.setItem(key, value);
        } catch {
            // Ignored in sandboxed iframes where storage access is restricted
        }
    };

    const VIEW_META = {
        analyzer: {
            section: "Moderation",
            page: "Live Text Analyzer",
            heading: "Live Text Analyzer",
            subheading: "Analyze social posts with dialect cleaning, fast-pass filtering, and calibrated toxicity scoring.",
        },
        review: {
            section: "Moderation",
            page: "Borderline Queue",
            heading: "Borderline Review Queue",
            subheading: "Inspect posts in the 60%–80% confidence band and record moderator decisions.",
        },
        monitor: {
            section: "Intelligence",
            page: "Hate-Raid Monitor",
            heading: "Hate-Raid & Spike Monitor",
            subheading: "Track real-time toxicity trajectories, category share, and thread-level spike alerts.",
        },
        logs: {
            section: "Intelligence",
            page: "Moderation Logs",
            heading: "Moderation Logs",
            subheading: "Inspect persisted moderation decisions and filter by action.",
        },
        policy: {
            section: "Configuration",
            page: "Threshold & Policy",
            heading: "Threshold & Policy",
            subheading: "Active moderation thresholds and loaded machine learning backends.",
        },
    };

    const state = {
        darkMode: readStorage("ss_theme", "dark") !== "light",
        activeView: "analyzer",
        sidebarCollapsed: false,
        logFilter: "",
        lastPrediction: null,
        lastLogs: [],
    };

    const escapeHtml = (value) =>
        String(value ?? "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");

    const pct = (val) => `${(Number(val || 0) * 100).toFixed(1)}%`;

    const actionPillClasses = (action) => {
        if (action === "allow") {
            return "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30";
        }
        if (action === "block") {
            return "bg-red-500/15 text-red-600 dark:text-red-400 border border-red-500/30";
        }
        return "bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30";
    };

    const barColor = (score) => {
        if (score >= 0.8) return "#ef4444";
        if (score >= 0.6) return "#f59e0b";
        return "#7c6cf6";
    };

    function setApiStatus(isOnline, reason = "") {
        const pill = document.getElementById("api-status-pill");
        const banner = document.getElementById("api-offline-banner");
        if (pill) pill.remove();
        if (banner) banner.remove();
    }

    async function apiFetch(path, options = {}) {
        const isBodyMethod = options.method && !["GET", "HEAD"].includes(options.method.toUpperCase());
        const headers = {
            Accept: "application/json",
            ...(isBodyMethod ? { "Content-Type": "application/json" } : {}),
            ...(options.headers || {}),
        };
        try {
            const resp = await fetch(`${API_BASE}${path}`, {
                ...options,
                headers,
            });
            if (!resp.ok) {
                const text = await resp.text();
                throw new Error(text || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            // Automatically mark API online upon ANY successful response
            setApiStatus(true);
            return data;
        } catch (err) {
            if (path.startsWith("/health")) {
                setApiStatus(false, err.message);
            }
            throw err;
        }
    }

    function applyTheme() {
        const root = document.documentElement;
        const icon = document.getElementById("theme-icon");
        const label = document.getElementById("theme-label");
        if (state.darkMode) {
            root.classList.add("dark");
            icon.textContent = "☀️";
            label.textContent = "Light";
            writeStorage("ss_theme", "dark");
        } else {
            root.classList.remove("dark");
            icon.textContent = "🌙";
            label.textContent = "Dark";
            writeStorage("ss_theme", "light");
        }
        if (state.lastPrediction) renderCategoryBarChart(state.lastPrediction.category_scores);
        if (state.lastLogs.length > 0) renderSpikeTimelineChart(state.lastLogs);
    }

    function openMobileSidebar() {
        const sidebar = document.getElementById("sidebar");
        const backdrop = document.getElementById("sidebar-backdrop");
        if (!sidebar) return;
        sidebar.classList.remove("-translate-x-full");
        if (backdrop) backdrop.classList.remove("hidden");
        document.body.classList.add("overflow-hidden");
    }

    function closeMobileSidebar() {
        const sidebar = document.getElementById("sidebar");
        const backdrop = document.getElementById("sidebar-backdrop");
        if (!sidebar) return;
        if (window.innerWidth < 768) {
            sidebar.classList.add("-translate-x-full");
        }
        if (backdrop) backdrop.classList.add("hidden");
        document.body.classList.remove("overflow-hidden");
    }

    function toggleSidebar() {
        if (window.innerWidth < 768) {
            closeMobileSidebar();
            return;
        }
        state.sidebarCollapsed = !state.sidebarCollapsed;
        const sidebar = document.getElementById("sidebar");
        const brandBar = document.getElementById("sidebar-brand-bar");
        sidebar.classList.toggle("md:w-64", !state.sidebarCollapsed);
        sidebar.classList.toggle("md:w-16", state.sidebarCollapsed);
        if (brandBar) {
            brandBar.classList.toggle("justify-between", !state.sidebarCollapsed);
            brandBar.classList.toggle("px-4", !state.sidebarCollapsed);
            brandBar.classList.toggle("justify-center", state.sidebarCollapsed);
            brandBar.classList.toggle("px-2", state.sidebarCollapsed);
        }
        sidebar.querySelectorAll(".sidebar-label").forEach((el) => {
            el.classList.toggle("hidden", state.sidebarCollapsed);
        });
    }

    async function checkHealth() {
        const policyPre = document.getElementById("policy-backends-json");
        try {
            const data = await apiFetch(`/health?_=${Date.now()}`);
            setApiStatus(true);
            if (policyPre) policyPre.textContent = JSON.stringify(data, null, 2);
            return true;
        } catch (err) {
            console.warn("SpeechShield API health check failed:", err);
            setApiStatus(false, err.message);
            if (policyPre) policyPre.textContent = `API offline: ${err.message || err}`;
            return false;
        }
    }

    function renderHorizontalMeters(containerId, entries) {
        const container = document.getElementById(containerId);
        container.innerHTML = entries
            .map(([label, rawScore]) => {
                const score = Math.max(0, Math.min(1, Number(rawScore || 0)));
                const color = barColor(score);
                return `
                    <div>
                        <div class="flex items-center justify-between text-xs mb-1">
                            <span class="font-medium text-gray-800 dark:text-gray-200">${escapeHtml(label)}</span>
                            <span class="font-mono tabular-nums text-gray-500 dark:text-[#8e95a5]">${pct(score)}</span>
                        </div>
                        <div class="w-full h-1.5 rounded-full bg-gray-200 dark:bg-white/[0.08] overflow-hidden">
                            <div class="h-full rounded-full transition-all duration-300" style="width: ${(score * 100).toFixed(1)}%; background: ${color};"></div>
                        </div>
                    </div>
                `;
            })
            .join("");
    }

    function renderCategoryBarChart(categoryScores) {
        const container = document.getElementById("category-bar-chart");
        const entries = Object.entries(categoryScores).sort((a, b) => b[1] - a[1]);
        const width = 560;
        const rowH = 34;
        const padLeft = 104;
        const padRight = 52;
        const padTop = 10;
        const padBottom = 26;
        const height = padTop + entries.length * rowH + padBottom;
        const plotW = width - padLeft - padRight;

        const gridColor = state.darkMode ? "#242733" : "#e5e7eb";
        const labelColor = state.darkMode ? "#8e95a5" : "#6b7280";
        const thresholdX = padLeft + 0.6 * plotW;

        const gridLines = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            .map((tick) => {
                const x = padLeft + tick * plotW;
                return `
                    <line x1="${x}" y1="${padTop}" x2="${x}" y2="${height - padBottom}" stroke="${gridColor}" stroke-width="1" />
                    <text x="${x}" y="${height - 8}" text-anchor="middle" fill="${labelColor}" font-size="10" font-family="JetBrains Mono, monospace">${Math.round(tick * 100)}%</text>
                `;
            })
            .join("");

        const bars = entries
            .map(([cat, val], idx) => {
                const score = Math.max(0, Math.min(1, Number(val || 0)));
                const y = padTop + idx * rowH + 6;
                const barW = Math.max(2, score * plotW);
                const color = barColor(score);
                const label = cat.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
                return `
                    <g>
                        <text x="${padLeft - 10}" y="${y + 13}" text-anchor="end" fill="${labelColor}" font-size="13" font-family="'Oregano', cursive, sans-serif">${escapeHtml(label)}</text>
                        <rect x="${padLeft}" y="${y}" width="${plotW}" height="18" rx="4" fill="${state.darkMode ? "rgba(255,255,255,0.04)" : "#f3f4f6"}" />
                        <rect x="${padLeft}" y="${y}" width="${barW}" height="18" rx="4" fill="${color}" />
                        <text x="${padLeft + plotW + 8}" y="${y + 13}" text-anchor="start" fill="${labelColor}" font-size="11" font-family="JetBrains Mono, monospace">${pct(score)}</text>
                    </g>
                `;
            })
            .join("");

        container.innerHTML = `
            <svg viewBox="0 0 ${width} ${height}" class="w-full h-auto overflow-visible">
                ${gridLines}
                ${bars}
                <line x1="${thresholdX}" y1="${padTop}" x2="${thresholdX}" y2="${height - padBottom}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.85" />
            </svg>
        `;
    }

    function renderSpikeTimelineChart(logs) {
        const container = document.getElementById("spike-timeline-chart");
        if (!logs || logs.length === 0) {
            container.innerHTML = `<div class="py-12 text-center text-xs text-gray-500 dark:text-[#8e95a5]">No moderation logs recorded yet. Analyze comments to populate the trajectory.</div>`;
            return;
        }

        const ordered = [...logs].reverse();
        const width = 680;
        const height = 240;
        const padLeft = 44;
        const padRight = 16;
        const padTop = 16;
        const padBottom = 28;
        const plotW = width - padLeft - padRight;
        const plotH = height - padTop - padBottom;

        const gridColor = state.darkMode ? "#242733" : "#e5e7eb";
        const labelColor = state.darkMode ? "#8e95a5" : "#6b7280";

        const coords = ordered.map((item, idx) => {
            const x = ordered.length === 1 ? padLeft + plotW / 2 : padLeft + (idx / (ordered.length - 1)) * plotW;
            const score = Math.max(0, Math.min(1, Number(item.overall_score || 0)));
            const y = padTop + (1 - score) * plotH;
            return { x, y, score, item, seq: idx + 1 };
        });

        const linePath = coords.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
        const areaPath = `${linePath} L ${coords[coords.length - 1].x.toFixed(1)} ${(padTop + plotH).toFixed(1)} L ${coords[0].x.toFixed(1)} ${(padTop + plotH).toFixed(1)} Z`;

        const yGrid = [0, 0.25, 0.5, 0.75, 1.0]
            .map((t) => {
                const y = padTop + (1 - t) * plotH;
                return `
                    <line x1="${padLeft}" y1="${y}" x2="${width - padRight}" y2="${y}" stroke="${gridColor}" stroke-width="1" />
                    <text x="${padLeft - 8}" y="${y + 3}" text-anchor="end" fill="${labelColor}" font-size="10" font-family="JetBrains Mono, monospace">${Math.round(t * 100)}%</text>
                `;
            })
            .join("");

        const y60 = padTop + (1 - 0.6) * plotH;
        const y80 = padTop + (1 - 0.8) * plotH;

        const circles = coords
            .map(
                (p) => `
                    <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4" fill="#8b7cf8" stroke="${state.darkMode ? "#14161d" : "#ffffff"}" stroke-width="1.5" class="timeline-pt cursor-pointer"
                        data-seq="${p.seq}" data-post="${escapeHtml(p.item.post_id)}" data-platform="${escapeHtml(p.item.platform)}" data-action="${escapeHtml(p.item.action)}" data-score="${pct(p.score)}" />
                `
            )
            .join("");

        container.innerHTML = `
            <div class="relative">
                <svg viewBox="0 0 ${width} ${height}" class="w-full h-auto overflow-visible">
                    <defs>
                        <linearGradient id="ssAreaGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stop-color="#7c6cf6" stop-opacity="0.34" />
                            <stop offset="100%" stop-color="#7c6cf6" stop-opacity="0.01" />
                        </linearGradient>
                    </defs>
                    ${yGrid}
                    <line x1="${padLeft}" y1="${y60}" x2="${width - padRight}" y2="${y60}" stroke="#f59e0b" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.8" />
                    <line x1="${padLeft}" y1="${y80}" x2="${width - padRight}" y2="${y80}" stroke="#ef4444" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.8" />
                    ${ordered.length > 1 ? `<path d="${areaPath}" fill="url(#ssAreaGrad)" /><path d="${linePath}" fill="none" stroke="#7c6cf6" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" />` : ""}
                    ${circles}
                </svg>
                <div id="timeline-tooltip" class="chart-tooltip hidden absolute z-10 px-2.5 py-1.5 rounded-lg text-[11px] bg-gray-900 text-gray-100 border border-white/10 shadow-lg pointer-events-none"></div>
            </div>
        `;

        const tooltip = document.getElementById("timeline-tooltip");
        container.querySelectorAll(".timeline-pt").forEach((el) => {
            el.addEventListener("mouseenter", (evt) => {
                const t = evt.target;
                tooltip.innerHTML = `<div class="font-semibold">#${t.dataset.seq} · ${t.dataset.post}</div><div class="text-gray-300 font-mono">${t.dataset.platform} · ${t.dataset.action} · ${t.dataset.score}</div>`;
                tooltip.classList.remove("hidden");
            });
            el.addEventListener("mousemove", (evt) => {
                const rect = container.getBoundingClientRect();
                tooltip.style.left = `${Math.min(rect.width - 170, evt.clientX - rect.left + 12)}px`;
                tooltip.style.top = `${Math.max(0, evt.clientY - rect.top - 36)}px`;
            });
            el.addEventListener("mouseleave", () => tooltip.classList.add("hidden"));
        });
    }

    function renderPrediction(data) {
        state.lastPrediction = data;
        document.getElementById("analyzer-results").classList.remove("hidden");

        const score = Number(data.overall_score || 0);
        const flagged = data.flagged_categories || [];
        const signals = data.signals || {};

        document.getElementById("finding-verdict-title").textContent = `Moderation Verdict: ${String(data.action).toUpperCase()} (${pct(score)})`;
        document.getElementById("finding-verdict-body").textContent = `Flagged dimensions: ${flagged.length ? flagged.join(", ") : "none"}. Evaluated on ${data.platform} thread ${data.post_id} by @${data.author_handle}.`;

        const verdictMap = {
            allow: "Safe · Allow",
            review: "Borderline · 60–80%",
            block: "High Toxicity · Block",
        };
        const pill = document.getElementById("finding-verdict-pill");
        pill.textContent = verdictMap[data.action] || data.action;
        pill.className = `px-2.5 py-1 rounded-full text-xs font-semibold ${actionPillClasses(data.action)}`;
        document.getElementById("finding-model-tag").textContent = data.model_used;

        document.getElementById("finding-cleaned-body").textContent = data.cleaned_text || "(empty after cleaning)";
        const sent = Number(signals.sentiment_score || 0);
        const sentPill = document.getElementById("finding-sentiment-pill");
        sentPill.textContent = `Sentiment ${sent >= 0 ? "+" : ""}${sent.toFixed(2)}`;
        sentPill.className = `px-2.5 py-1 rounded-full text-xs font-semibold tabular-nums ${
            sent < -0.3
                ? "bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30"
                : "bg-accent-500/15 text-accent-600 dark:text-accent-400 border border-accent-500/30"
        }`;
        document.getElementById("finding-obfuscation-tag").textContent = `Obfuscations: ${signals.obfuscation_hits || 0}`;

        const dimBadge = document.getElementById("dimension-verdict-badge");
        dimBadge.textContent = `${data.action.toUpperCase()} · ${pct(score)}`;
        dimBadge.className = `px-2.5 py-1 rounded-full text-xs font-semibold tabular-nums ${actionPillClasses(data.action)}`;

        renderCategoryBarChart(data.category_scores || {});
        const sortedEntries = Object.entries(data.category_scores || {})
            .sort((a, b) => b[1] - a[1])
            .map(([k, v]) => [k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()), v]);
        renderHorizontalMeters("category-meters", sortedEntries);

        const cues = [
            ...(signals.hashtags || []).map((h) => `#${h}`),
            ...(signals.emojis || []),
        ];
        const cuesWrapper = document.getElementById("extracted-cues-wrapper");
        const cuesEl = document.getElementById("extracted-cues");
        if (cues.length > 0) {
            cuesWrapper.classList.remove("hidden");
            cuesEl.innerHTML = cues
                .map((c) => `<span class="px-2 py-0.5 rounded-md text-xs font-mono bg-gray-100 dark:bg-[#1c1f29] border border-gray-200 dark:border-white/[0.08]">${escapeHtml(c)}</span>`)
                .join("");
        } else {
            cuesWrapper.classList.add("hidden");
        }
    }

    async function loadReviewQueue() {
        const listEl = document.getElementById("review-queue-list");
        const badgeEl = document.getElementById("review-count-badge");
        try {
            const queue = await apiFetch("/review-queue");
            if (queue.length > 0) {
                badgeEl.textContent = String(queue.length);
                badgeEl.classList.remove("hidden");
            } else {
                badgeEl.classList.add("hidden");
            }

            if (queue.length === 0) {
                listEl.innerHTML = `<div class="rounded-xl bg-white dark:bg-[#14161d] border border-gray-200 dark:border-white/[0.08] p-8 text-center text-sm text-gray-500 dark:text-[#8e95a5]">Review queue is empty. Borderline comments (60%–80%) will appear here automatically.</div>`;
                return;
            }

            listEl.innerHTML = queue
                .map(
                    (item) => `
                    <div class="rounded-xl bg-white dark:bg-[#14161d] border border-gray-200 dark:border-white/[0.08] p-4 sm:p-5 space-y-3" data-review-id="${item.id}">
                        <div class="flex flex-wrap items-center justify-between gap-2">
                            <div class="flex items-center gap-2 text-xs font-mono">
                                <span class="px-2 py-0.5 rounded-md bg-gray-100 dark:bg-[#1c1f29] border border-gray-200 dark:border-white/[0.08]">${escapeHtml(item.platform)}</span>
                                <span class="font-semibold text-gray-900 dark:text-gray-100">@${escapeHtml(item.author_handle)}</span>
                                <span class="text-gray-400">·</span>
                                <span class="text-gray-500 dark:text-[#8e95a5]">${escapeHtml(item.post_id)}</span>
                            </div>
                            <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold tabular-nums bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/30">${pct(item.overall_score)} Confidence</span>
                        </div>
                        <p class="text-sm text-gray-900 dark:text-gray-100">${escapeHtml(item.raw_text)}</p>
                        <div class="text-xs font-mono text-gray-500 dark:text-[#8e95a5]">Cleaned: ${escapeHtml(item.cleaned_text)}</div>
                        <div class="flex flex-wrap items-center gap-1.5">
                            ${(item.flagged_categories || []).map((c) => `<span class="px-2 py-0.5 rounded-md text-[11px] font-mono bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">${escapeHtml(c)}</span>`).join("")}
                        </div>
                        <div class="flex flex-col sm:flex-row gap-2 pt-2 border-t border-gray-100 dark:border-white/[0.06]">
                            <input type="text" placeholder="Optional moderator notes..." class="review-note-input flex-1 rounded-lg bg-gray-50 dark:bg-[#0c0d12] border border-gray-200 dark:border-white/[0.08] px-3 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-accent-500/50" />
                            <button type="button" class="review-approve-btn px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 transition" data-id="${item.id}">Approve Safe</button>
                            <button type="button" class="review-block-btn px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-red-500/15 hover:bg-red-500/25 text-red-600 dark:text-red-400 border border-red-500/30 transition" data-id="${item.id}">Confirm Toxic</button>
                        </div>
                    </div>
                `
                )
                .join("");

            listEl.querySelectorAll(".review-approve-btn, .review-block-btn").forEach((btn) => {
                btn.addEventListener("click", async () => {
                    const id = btn.dataset.id;
                    const card = btn.closest("[data-review-id]");
                    const note = card.querySelector(".review-note-input").value.trim();
                    const decision = btn.classList.contains("review-approve-btn") ? "approved_safe" : "confirmed_toxic";
                    const buttons = card.querySelectorAll("button");
                    buttons.forEach((b) => {
                        b.disabled = true;
                    });
                    try {
                        await apiFetch(`/review/${id}`, {
                            method: "POST",
                            body: JSON.stringify({ decision, moderator_notes: note || null }),
                        });
                        await Promise.all([loadReviewQueue(), loadMonitor()]);
                    } catch (err) {
                        alert(`Review action failed: ${err.message}`);
                        buttons.forEach((b) => {
                            b.disabled = false;
                        });
                    }
                });
            });
        } catch {
            listEl.innerHTML = `<div class="rounded-xl bg-white dark:bg-[#14161d] border border-gray-200 dark:border-white/[0.08] p-6 text-center text-xs text-gray-500 dark:text-[#8e95a5]">Connect the FastAPI backend to manage the review queue.</div>`;
        }
    }

    async function loadLogs() {
        try {
            const query = state.logFilter ? `?action=${encodeURIComponent(state.logFilter)}&limit=100` : "?limit=100";
            const logs = await apiFetch(`/logs${query}`);
            document.getElementById("logs-count-pill").textContent = `${logs.length} rows`;
            document.getElementById("logs-tbody").innerHTML = logs.length
                ? logs
                      .map(
                          (row) => `
                          <tr class="hover:bg-gray-50 dark:hover:bg-white/[0.02]">
                              <td class="py-2.5 px-3 font-mono text-gray-500 dark:text-[#8e95a5] whitespace-nowrap">#${row.id}</td>
                              <td class="py-2.5 px-3 font-mono whitespace-nowrap">${escapeHtml(row.platform)}</td>
                              <td class="py-2.5 px-3 font-mono text-gray-500 dark:text-[#8e95a5] whitespace-nowrap">${escapeHtml(row.post_id)}</td>
                              <td class="py-2.5 px-3 font-mono whitespace-nowrap">@${escapeHtml(row.author_handle)}</td>
                              <td class="py-2.5 px-3 max-w-[260px] truncate" title="${escapeHtml(row.raw_text)}">${escapeHtml(row.raw_text)}</td>
                              <td class="py-2.5 px-3 font-mono tabular-nums whitespace-nowrap">${pct(row.overall_score)}</td>
                              <td class="py-2.5 px-3 whitespace-nowrap"><span class="px-2 py-0.5 rounded-full text-[11px] font-semibold ${actionPillClasses(row.action)}">${escapeHtml(row.action)}</span></td>
                              <td class="py-2.5 px-3 font-mono text-[11px] text-gray-500 dark:text-[#8e95a5] whitespace-nowrap">${escapeHtml(row.review_status)}</td>
                          </tr>
                      `
                      )
                      .join("")
                : `<tr><td colspan="8" class="py-8 text-center text-xs text-gray-500 dark:text-[#8e95a5]">No logs found for the selected filter.</td></tr>`;
        } catch {
            document.getElementById("logs-tbody").innerHTML = "";
        }
    }

    async function loadMonitor() {
        try {
            const [stats, logs] = await Promise.all([apiFetch("/analytics/spikes"), apiFetch("/logs?limit=100")]);
            state.lastLogs = logs;

            const total = Number(stats.total_processed || 0);
            const toxic = Number(stats.toxic_count || 0);
            const blocked = Number(stats.blocked_count || 0);
            const pending = Number(stats.pending_review_count || 0);
            const avg = Number(stats.avg_toxicity_score || 0);
            const alerts = stats.hate_raid_alerts || [];

            const alertBadge = document.getElementById("alerts-count-badge");
            if (alerts.length > 0) {
                alertBadge.textContent = String(alerts.length);
                alertBadge.classList.remove("hidden");
            } else {
                alertBadge.classList.add("hidden");
            }

            document.getElementById("kpi-total").textContent = String(total);
            document.getElementById("kpi-toxic").textContent = String(toxic);
            document.getElementById("kpi-toxic-pill").textContent = total ? `${Math.round((toxic / total) * 100)}%` : "0%";
            document.getElementById("kpi-blocked").textContent = String(blocked);
            document.getElementById("kpi-pending").textContent = String(pending);
            document.getElementById("avg-score-pill").textContent = `Avg ${pct(avg)}`;

            if (alerts.length > 0) {
                const top = alerts[0];
                document.getElementById("spike-finding-icon").textContent = "🚨";
                document.getElementById("spike-finding-title").textContent = `${alerts.length} Active Hate-Raid Alert(s) Detected`;
                document.getElementById("spike-finding-body").textContent = `Thread ${top.post_id} shows ${Math.round(top.toxic_ratio * 100)}% toxic replies (${top.toxic_comments}/${top.total_comments} comments).`;
                const p = document.getElementById("spike-finding-pill");
                p.textContent = `${String(top.alert_level).toUpperCase()} SPIKE`;
                p.className = "px-2.5 py-1 rounded-full text-xs font-semibold bg-red-500/15 text-red-600 dark:text-red-400 border border-red-500/30";
                document.getElementById("spike-finding-tag").textContent = `Post: ${top.post_id}`;

                document.getElementById("hate-raid-card").classList.remove("hidden");
                document.getElementById("hate-raid-tbody").innerHTML = alerts
                    .map(
                        (a) => `
                        <tr>
                            <td class="py-2 pr-4 font-mono font-medium">${escapeHtml(a.post_id)}</td>
                            <td class="py-2 pr-4 font-mono">${a.toxic_comments} / ${a.total_comments}</td>
                            <td class="py-2 pr-4 font-mono">${pct(a.toxic_ratio)}</td>
                            <td class="py-2 pr-4 font-mono">${pct(a.avg_toxicity_score)}</td>
                            <td class="py-2"><span class="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-red-500/15 text-red-600 dark:text-red-400">${escapeHtml(a.alert_level)}</span></td>
                        </tr>
                    `
                    )
                    .join("");
            } else {
                document.getElementById("spike-finding-icon").textContent = "🛡️";
                document.getElementById("spike-finding-title").textContent = "Social Feeds Nominal";
                document.getElementById("spike-finding-body").textContent = "No coordinated hate-raid bursts exceed the 50% thread toxicity threshold.";
                const p = document.getElementById("spike-finding-pill");
                p.textContent = "Healthy";
                p.className = "px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30";
                document.getElementById("spike-finding-tag").textContent = `Avg Score: ${pct(avg)}`;
                document.getElementById("hate-raid-card").classList.add("hidden");
            }

            const catEntries = Object.entries(stats.category_counts || {});
            const [topCat, topCatCount] = catEntries.sort((a, b) => b[1] - a[1])[0] || ["none", 0];
            document.getElementById("vector-finding-title").textContent = `Top Flagged Vector: ${topCat.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}`;
            document.getElementById("vector-finding-body").textContent = `Recorded ${topCatCount} hits across ${total} evaluated comments with ${pending} borderline items queued for review.`;
            document.getElementById("vector-finding-pill").textContent = total ? `${pct(toxic / total)} Toxic Rate` : "0.0% Toxic Rate";

            renderSpikeTimelineChart(logs);
            const denom = Math.max(total, 1);
            renderHorizontalMeters(
                "monitor-category-meters",
                catEntries
                    .map(([k, v]) => [k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()), Number(v) / denom])
                    .sort((a, b) => b[1] - a[1])
            );
        } catch {
            renderSpikeTimelineChart([]);
        }
    }

    async function switchView(viewName, options = {}) {
        closeMobileSidebar();
        state.activeView = viewName;
        const meta = VIEW_META[viewName] || VIEW_META.analyzer;
        document.getElementById("breadcrumb-section").textContent = meta.section;
        document.getElementById("breadcrumb-page").textContent = meta.page;
        document.getElementById("page-heading").textContent = meta.heading;
        document.getElementById("page-subheading").textContent = meta.subheading;

        document.querySelectorAll(".nav-btn").forEach((btn) => {
            const active = btn.dataset.view === viewName;
            btn.className = active
                ? "nav-btn w-full flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg text-[13px] font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60 transition text-left bg-accent-500/15 text-accent-600 dark:text-[#b4a9ff] border border-accent-500/25"
                : "nav-btn w-full flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg text-[13px] font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-500/60 transition text-left text-gray-600 dark:text-[#8e95a5] hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-white/[0.04] border border-transparent";
        });

        document.querySelectorAll(".view-panel").forEach((panel) => {
            panel.classList.toggle("hidden", panel.id !== `view-${viewName}`);
        });

        if (viewName === "review") await loadReviewQueue();
        if (viewName === "monitor") {
            await loadMonitor();
            if (options.scrollTo) {
                const target = document.getElementById(options.scrollTo);
                if (target && !target.classList.contains("hidden")) {
                    target.scrollIntoView({ behavior: "smooth", block: "start" });
                }
            }
        }
        if (viewName === "logs") await loadLogs();
        if (viewName === "policy") await checkHealth();
    }

    document.addEventListener("DOMContentLoaded", async () => {
        applyTheme();
        switchView("analyzer");

        document.getElementById("theme-toggle-btn").addEventListener("click", () => {
            state.darkMode = !state.darkMode;
            applyTheme();
        });

        document.getElementById("sidebar-collapse-btn")?.addEventListener("click", toggleSidebar);
        document.getElementById("mobile-sidebar-toggle")?.addEventListener("click", openMobileSidebar);
        document.getElementById("sidebar-close-btn")?.addEventListener("click", closeMobileSidebar);
        document.getElementById("sidebar-backdrop")?.addEventListener("click", closeMobileSidebar);

        window.addEventListener("resize", () => {
            if (window.innerWidth >= 768) {
                closeMobileSidebar();
            }
        });

        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                closeMobileSidebar();
            }
        });

        document.querySelectorAll("[data-view]").forEach((btn) => {
            btn.addEventListener("click", () => {
                switchView(btn.dataset.view, { scrollTo: btn.dataset.scroll || null });
            });
        });

        document.querySelectorAll(".log-filter-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                state.logFilter = btn.dataset.filter || "";
                document.querySelectorAll(".log-filter-btn").forEach((b) => {
                    const active = (b.dataset.filter || "") === state.logFilter;
                    b.className = active
                        ? "log-filter-btn px-3 py-1 rounded-lg text-xs font-medium bg-accent-500/15 text-accent-600 dark:text-accent-400 border border-accent-500/30"
                        : "log-filter-btn px-3 py-1 rounded-lg text-xs font-medium bg-gray-100 dark:bg-[#1c1f29] text-gray-600 dark:text-[#8e95a5] border border-gray-200 dark:border-white/[0.08]";
                });
                loadLogs();
            });
        });

        document.querySelectorAll(".sample-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const input = document.getElementById("comment-input");
                input.value = btn.dataset.sample;
                input.focus();
            });
        });

        document.getElementById("analyze-form").addEventListener("submit", async (e) => {
            e.preventDefault();
            const text = document.getElementById("comment-input").value.trim();
            if (!text) return;
            const submitBtn = document.getElementById("analyze-submit-btn");
            submitBtn.disabled = true;
            submitBtn.textContent = "Analyzing...";
            try {
                const payload = {
                    text,
                    platform: document.getElementById("platform-select").value,
                    post_id: document.getElementById("post-id-input").value.trim() || "post_101",
                    author_handle: document.getElementById("author-input").value.trim() || "social_user",
                };
                const result = await apiFetch("/predict", { method: "POST", body: JSON.stringify(payload) });
                renderPrediction(result);
                loadReviewQueue();
                loadMonitor();
            } catch (err) {
                alert(`Prediction failed: ${err.message}`);
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = "Analyze Comment";
            }
        });

        document.getElementById("refresh-review-btn").addEventListener("click", loadReviewQueue);
        document.getElementById("refresh-monitor-btn").addEventListener("click", loadMonitor);
        document.getElementById("refresh-logs-btn").addEventListener("click", loadLogs);

        const retryBtn = document.getElementById("retry-health-btn");
        if (retryBtn) {
            retryBtn.addEventListener("click", async () => {
                retryBtn.disabled = true;
                const originalText = retryBtn.textContent;
                retryBtn.textContent = "Checking...";
                try {
                    const online = await checkHealth();
                    if (online) {
                        loadReviewQueue();
                        loadMonitor();
                        if (state.activeView === "logs") loadLogs();
                    }
                } finally {
                    retryBtn.disabled = false;
                    retryBtn.textContent = originalText;
                }
            });
        }

        function showWorkspace() {
            const landing = document.getElementById("landing-hero");
            const workspace = document.getElementById("app-workspace");
            if (landing) landing.style.display = "none";
            if (workspace) workspace.style.display = "flex";
            try {
                window.location.hash = "app";
            } catch {}
            checkHealth().then((online) => {
                if (online) {
                    loadReviewQueue();
                    loadMonitor();
                    if (state.activeView === "logs") loadLogs();
                }
            });
        }

        function showLanding() {
            closeMobileSidebar();
            const landing = document.getElementById("landing-hero");
            const workspace = document.getElementById("app-workspace");
            if (workspace) workspace.style.display = "none";
            if (landing) landing.style.display = "flex";
            try {
                history.replaceState(null, "", window.location.pathname + window.location.search);
            } catch {}
        }

        // Attach Landing <-> Workspace transitions
        document.querySelectorAll(".enter-app-trigger").forEach((el) => {
            el.addEventListener("click", (e) => {
                e.preventDefault();
                showWorkspace();
            });
        });

        const exitLandingBtn = document.getElementById("exit-to-landing-btn");
        if (exitLandingBtn) {
            exitLandingBtn.addEventListener("click", showLanding);
        }

        const brandLogo = document.getElementById("sidebar-brand-logo");
        if (brandLogo) {
            brandLogo.addEventListener("click", showLanding);
        }

        const statusPill = document.getElementById("api-status-pill");
        if (statusPill) {
            statusPill.remove();
        }
        const offlineBanner = document.getElementById("api-offline-banner");
        if (offlineBanner) {
            offlineBanner.remove();
        }

        // Interactive 3D Cosmic Parallax for planets and celestial bodies
        const heroSection = document.getElementById("landing-hero");
        const parallaxLayers = document.querySelectorAll(".cosmic-parallax-layer");
        if (heroSection && parallaxLayers.length > 0) {
            let targetX = 0;
            let targetY = 0;
            let currentX = 0;
            let currentY = 0;
            let rafId = null;

            function updateParallax() {
                currentX += (targetX - currentX) * 0.08;
                currentY += (targetY - currentY) * 0.08;

                parallaxLayers.forEach((layer) => {
                    const depth = parseFloat(layer.dataset.depth || "0.05");
                    const moveX = (currentX * depth * 140).toFixed(2);
                    const moveY = (currentY * depth * 140).toFixed(2);
                    layer.style.transform = `translate3d(${moveX}px, ${moveY}px, 0)`;
                });

                if (Math.abs(targetX - currentX) > 0.0005 || Math.abs(targetY - currentY) > 0.0005) {
                    rafId = requestAnimationFrame(updateParallax);
                } else {
                    rafId = null;
                }
            }

            function onMouseMove(e) {
                const rect = heroSection.getBoundingClientRect();
                targetX = (e.clientX - rect.left) / rect.width - 0.5;
                targetY = (e.clientY - rect.top) / rect.height - 0.5;
                if (!rafId) {
                    rafId = requestAnimationFrame(updateParallax);
                }
            }

            heroSection.addEventListener("mousemove", onMouseMove, { passive: true });
            if (window.DeviceOrientationEvent) {
                window.addEventListener("deviceorientation", (e) => {
                    if (e.gamma != null && e.beta != null) {
                        targetX = Math.max(-0.5, Math.min(0.5, e.gamma / 60));
                        targetY = Math.max(-0.5, Math.min(0.5, (e.beta - 45) / 60));
                        if (!rafId) {
                            rafId = requestAnimationFrame(updateParallax);
                        }
                    }
                }, { passive: true });
            }
        }

        // Route on initial load: open workspace if #app or ?view=app, else show landing hero
        if (window.location.hash === "#app" || params.get("view") === "app") {
            showWorkspace();
        } else {
            showLanding();
        }

        const online = await checkHealth();
        if (online) {
            loadReviewQueue();
            loadMonitor();
        } else {
            // Auto-poll every 5 seconds (up to 6 times) while the Render free-tier instance finishes waking up
            let attempts = 0;
            const poller = setInterval(async () => {
                attempts += 1;
                const ok = await checkHealth();
                if (ok || attempts >= 6) {
                    clearInterval(poller);
                    if (ok) {
                        loadReviewQueue();
                        loadMonitor();
                        if (state.activeView === "logs") loadLogs();
                    }
                }
            }, 5000);
        }
    });
})();
