/* stocklab web UI: theme, charts, interaction.
 *
 * No chart library. Everything is inline SVG built from a data attribute,
 * which keeps the app dependency-free and means the markup degrades to a
 * readable table when JavaScript is off -- every chart on these pages has the
 * same numbers in a table beside or beneath it.
 */
(function () {
  "use strict";

  /* ---------------- theme ---------------- */
  var STORE_KEY = "stocklab-theme";

  function readStored() {
    // Browser storage can throw in a private window or with site data blocked,
    // so every access is guarded and the page renders fine without it.
    try { return localStorage.getItem(STORE_KEY); } catch (e) { return null; }
  }
  function writeStored(value) {
    try { localStorage.setItem(STORE_KEY, value); } catch (e) { /* ignore */ }
  }

  function applyTheme(theme) {
    if (theme === "dark" || theme === "light") {
      document.documentElement.setAttribute("data-theme", theme);
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    var btn = document.querySelector(".theme-toggle");
    if (btn) {
      var dark = theme === "dark" || (theme !== "light" &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
      btn.textContent = dark ? "☀" : "☽";
      btn.setAttribute("aria-label", dark ? "Switch to light theme"
                                          : "Switch to dark theme");
    }
  }

  function currentTheme() {
    var stored = readStored();
    if (stored) return stored;
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark" : "light";
  }

  /* ---------------- formatting ---------------- */
  function fmtMoney(v) {
    if (v === null || v === undefined || isNaN(v)) return "–";
    var sign = v < 0 ? "-" : "", a = Math.abs(v);
    if (a >= 1e12) return sign + "$" + (a / 1e12).toFixed(2) + "T";
    if (a >= 1e9) return sign + "$" + (a / 1e9).toFixed(2) + "B";
    if (a >= 1e6) return sign + "$" + (a / 1e6).toFixed(2) + "M";
    if (a >= 1e3) return sign + "$" + (a / 1e3).toFixed(1) + "K";
    return sign + "$" + a.toFixed(2);
  }
  function fmtPrice(v) {
    if (v === null || v === undefined || isNaN(v)) return "–";
    return "$" + v.toLocaleString(undefined,
      { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  /* ---------------- tooltip ---------------- */
  var tip = null;
  function tooltip() {
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "tooltip";
      document.body.appendChild(tip);
    }
    return tip;
  }
  function showTip(html, x, y) {
    var t = tooltip();
    t.innerHTML = html;
    t.classList.add("on");
    var rect = t.getBoundingClientRect();
    var left = Math.min(x + 12, window.innerWidth - rect.width - 8);
    var top = Math.max(8, y - rect.height - 10);
    t.style.left = left + "px";
    t.style.top = top + "px";
  }
  function hideTip() { if (tip) tip.classList.remove("on"); }

  /* ---------------- line chart ---------------- */
  function drawLine(host) {
    var payload;
    try { payload = JSON.parse(host.getAttribute("data-series")); }
    catch (e) { return; }
    if (!payload || !payload.values || payload.values.length < 2) return;

    var values = payload.values, dates = payload.dates || [];
    var W = host.clientWidth || 640;
    var H = parseInt(host.getAttribute("data-height") || "150", 10);
    var padL = 46, padR = 10, padT = 8, padB = 18;
    var innerW = Math.max(10, W - padL - padR);
    var innerH = Math.max(10, H - padT - padB);

    var lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
    if (hi === lo) { hi = lo + 1; }
    var span = hi - lo;
    lo -= span * 0.06; hi += span * 0.06; span = hi - lo;

    function sx(i) { return padL + (i / (values.length - 1)) * innerW; }
    function sy(v) { return padT + (1 - (v - lo) / span) * innerH; }

    var svg = ['<svg class="chart" viewBox="0 0 ' + W + " " + H +
               '" width="100%" height="' + H + '" preserveAspectRatio="none" ' +
               'role="img" aria-label="Price history">'];

    // Recessive gridlines with direct value labels (no separate axis line).
    var ticks = 4;
    for (var g = 0; g <= ticks; g++) {
      var v = lo + (span * g) / ticks, y = sy(v);
      svg.push('<line class="gridline" x1="' + padL + '" y1="' + y +
               '" x2="' + (W - padR) + '" y2="' + y + '"/>');
      svg.push('<text x="' + (padL - 6) + '" y="' + (y + 3.5) +
               '" text-anchor="end">' + fmtAxis(v) + "</text>");
    }

    var d = "", area = "";
    for (var i = 0; i < values.length; i++) {
      d += (i ? "L" : "M") + sx(i).toFixed(1) + " " + sy(values[i]).toFixed(1);
    }
    area = d + "L" + sx(values.length - 1).toFixed(1) + " " + (padT + innerH) +
           "L" + padL + " " + (padT + innerH) + "Z";
    svg.push('<path class="area" d="' + area + '"/>');
    svg.push('<path class="line" d="' + d + '"/>');
    svg.push('<line class="crosshair" x1="0" y1="' + padT + '" x2="0" y2="' +
             (padT + innerH) + '" style="opacity:0"/>');
    svg.push('<circle class="dot" r="4" cx="0" cy="0" style="opacity:0"/>');

    // End label, so the latest value is readable without hovering.
    var lastX = sx(values.length - 1), lastY = sy(values[values.length - 1]);
    svg.push('<text x="' + (lastX - 2) + '" y="' + (lastY - 7) +
             '" text-anchor="end" style="fill:var(--series-1);font-weight:600">' +
             fmtAxis(values[values.length - 1]) + "</text>");
    svg.push("</svg>");
    host.innerHTML = svg.join("");

    var svgEl = host.querySelector("svg");
    var cross = svgEl.querySelector(".crosshair");
    var dot = svgEl.querySelector(".dot");

    function move(ev) {
      var box = svgEl.getBoundingClientRect();
      var px = ((ev.clientX - box.left) / box.width) * W;
      var idx = Math.round(((px - padL) / innerW) * (values.length - 1));
      idx = Math.max(0, Math.min(values.length - 1, idx));
      var x = sx(idx), y = sy(values[idx]);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x);
      cross.style.opacity = 0.55;
      dot.setAttribute("cx", x); dot.setAttribute("cy", y);
      dot.style.opacity = 1;
      var first = values[0], pct = first ? (values[idx] / first - 1) * 100 : NaN;
      showTip("<strong>" + fmtPrice(values[idx]) + "</strong>" +
              (dates[idx] ? '<br><span style="color:var(--text-muted)">' +
                            dates[idx] + "</span>" : "") +
              (isNaN(pct) ? "" : "<br>" + (pct >= 0 ? "+" : "") +
                            pct.toFixed(1) + "% from start"),
              ev.clientX, ev.clientY);
    }
    function leave() {
      cross.style.opacity = 0; dot.style.opacity = 0; hideTip();
    }
    svgEl.addEventListener("mousemove", move);
    svgEl.addEventListener("mouseleave", leave);
    svgEl.addEventListener("touchmove", function (ev) {
      if (ev.touches[0]) move(ev.touches[0]);
    }, { passive: true });
    svgEl.addEventListener("touchend", leave);
  }

  function fmtAxis(v) {
    var a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(1) + "B";
    if (a >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (a >= 1e3) return (v / 1e3).toFixed(0) + "k";
    if (a >= 100) return v.toFixed(0);
    return v.toFixed(a < 10 ? 2 : 1);
  }

  /* ---------------- percentile bands ---------------- */
  function placeBands() {
    document.querySelectorAll(".band[data-band]").forEach(function (el) {
      var d;
      try { d = JSON.parse(el.getAttribute("data-band")); } catch (e) { return; }
      var lo = d.p10, hi = d.p90;
      if (lo === null || hi === null || hi <= lo) return;
      // Keep the current reading inside the drawn range even when it sits
      // outside the historical 10-90 band.
      lo = Math.min(lo, d.current); hi = Math.max(hi, d.current);
      var span = hi - lo;
      function pos(v) {
        return Math.max(0, Math.min(100, ((v - lo) / span) * 100));
      }
      var iqr = el.querySelector(".iqr");
      if (iqr && d.p25 !== null && d.p75 !== null) {
        iqr.style.left = pos(d.p25) + "%";
        iqr.style.width = Math.max(1, pos(d.p75) - pos(d.p25)) + "%";
      }
      var med = el.querySelector(".median");
      if (med && d.median !== null) med.style.left = pos(d.median) + "%";
      var now = el.querySelector(".now");
      if (now) now.style.left = pos(d.current) + "%";
      el.setAttribute("title",
        "now " + d.current.toFixed(1) + "x · median " +
        (d.median === null ? "?" : d.median.toFixed(1)) + "x · range " +
        d.p10.toFixed(1) + "–" + d.p90.toFixed(1) + "x");
    });
  }

  /* ---------------- meters ---------------- */
  function placeMeters() {
    document.querySelectorAll("[data-fill]").forEach(function (el) {
      var pct = parseFloat(el.getAttribute("data-fill"));
      if (isNaN(pct)) return;
      var bar = el.querySelector("span");
      if (bar) bar.style.width = Math.max(0, Math.min(100, pct)) + "%";
    });
  }

  /* ---------------- sortable tables ---------------- */
  function makeSortable() {
    document.querySelectorAll("table[data-sortable]").forEach(function (table) {
      var head = table.querySelectorAll("thead th");
      head.forEach(function (th, index) {
        th.style.cursor = "pointer";
        th.title = "Click to sort";
        th.addEventListener("click", function () {
          var body = table.tBodies[0];
          if (!body) return;
          var rows = Array.prototype.slice.call(body.rows);
          var asc = th.getAttribute("data-dir") !== "asc";
          head.forEach(function (o) { o.removeAttribute("data-dir"); });
          th.setAttribute("data-dir", asc ? "asc" : "desc");
          rows.sort(function (a, b) {
            var x = cellValue(a, index), y = cellValue(b, index);
            if (typeof x === "number" && typeof y === "number") {
              return asc ? x - y : y - x;
            }
            return asc ? String(x).localeCompare(String(y))
                       : String(y).localeCompare(String(x));
          });
          rows.forEach(function (r) { body.appendChild(r); });
        });
      });
    });
  }
  function cellValue(row, index) {
    var cell = row.cells[index];
    if (!cell) return "";
    var raw = cell.getAttribute("data-sort");
    if (raw !== null && raw !== "") {
      var n = parseFloat(raw);
      return isNaN(n) ? raw : n;
    }
    var text = cell.textContent.trim().replace(/[$,%x\s]/g, "");
    if (text === "–" || text === "-" || text === "") return -Infinity;
    var num = parseFloat(text.replace(/[A-Za-z]+$/, ""));
    if (isNaN(num)) return cell.textContent.trim();
    var suffix = text.slice(-1);
    if (suffix === "T") num *= 1e12;
    else if (suffix === "B") num *= 1e9;
    else if (suffix === "M") num *= 1e6;
    else if (suffix === "K") num *= 1e3;
    return num;
  }

  /* ---------------- boot ---------------- */
  applyTheme(readStored());
  document.addEventListener("DOMContentLoaded", function () {
    applyTheme(readStored());
    var btn = document.querySelector(".theme-toggle");
    if (btn) {
      btn.addEventListener("click", function () {
        var next = currentTheme() === "dark" ? "light" : "dark";
        writeStored(next);
        applyTheme(next);
        document.querySelectorAll("[data-series]").forEach(drawLine);
      });
    }
    document.querySelectorAll("[data-series]").forEach(drawLine);
    placeBands();
    placeMeters();
    makeSortable();

    var resizeTimer = null;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(function () {
        document.querySelectorAll("[data-series]").forEach(drawLine);
      }, 150);
    });
  });
})();
