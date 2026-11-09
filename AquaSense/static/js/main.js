/* =========================================================
   AquaSense — Interaksi UI
   ========================================================= */
(function () {
  "use strict";

  /* ---------- Navigasi mobile ---------- */
  const navToggle = document.querySelector(".nav-toggle");
  const nav = document.getElementById("primary-nav");
  if (navToggle && nav) {
    navToggle.addEventListener("click", function () {
      const open = nav.classList.toggle("is-open");
      navToggle.setAttribute("aria-expanded", String(open));
    });
    nav.querySelectorAll("a").forEach(function (link) {
      link.addEventListener("click", function () {
        nav.classList.remove("is-open");
        navToggle.setAttribute("aria-expanded", "false");
      });
    });
  }

  /* ---------- Toast ---------- */
  const toast = document.getElementById("toast");
  let toastTimer = null;
  function showToast(message, type) {
    if (!toast) return;
    toast.textContent = message;
    toast.className = "toast is-visible" + (type ? " is-" + type : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toast.classList.remove("is-visible");
    }, 3200);
  }
  window.showToast = showToast;

  /* ---------- Sinkronisasi slider dan input angka ---------- */
  const numberInputs = document.querySelectorAll('.field-grid input[type="number"]');
  numberInputs.forEach(function (input) {
    const range = document.querySelector('[data-range-for="' + input.id + '"]');
    const counter = input.closest(".field")?.querySelector(".range-value");

    function format(value) {
      const decimals = (input.step.split(".")[1] || "").length;
      return Number(value).toFixed(decimals);
    }

    function update(value, source) {
      const min = parseFloat(input.min);
      const max = parseFloat(input.max);
      let next = parseFloat(value);
      if (Number.isNaN(next)) return;
      next = Math.min(max, Math.max(min, next));
      if (source !== "input") input.value = format(next);
      if (range && source !== "range") range.value = next;
      if (counter) counter.textContent = format(next);
    }

    input.addEventListener("input", function () {
      if (input.value !== "") update(input.value, "input");
    });
    input.addEventListener("blur", function () {
      if (input.value === "") return;
      input.value = input.value.replace(",", ".");
      update(input.value, "number");
    });
    if (range) {
      range.addEventListener("input", function () {
        update(range.value, "range");
      });
    }
  });

  /* ---------- Form prediksi ---------- */
  const predictForm = document.getElementById("predict-form");
  if (predictForm) {
    predictForm.addEventListener("submit", function (event) {
      const fields = predictForm.querySelectorAll('.field-grid input[type="number"]');
      let valid = true;
      fields.forEach(function (input) {
        input.value = (input.value || "").replace(",", ".").trim();
        if (!input.value || Number.isNaN(parseFloat(input.value))) {
          valid = false;
          input.setCustomValidity("Masukkan angka yang valid.");
          input.reportValidity();
        } else {
          input.setCustomValidity("");
        }
      });
      if (!valid) {
        event.preventDefault();
        return;
      }

      const button = document.getElementById("submit-btn");
      if (button) {
        button.disabled = true;
        button.classList.add("is-loading");
        const label = button.querySelector(".btn-label");
        if (label) label.textContent = "Menganalisis…";
      }
    });

    predictForm.addEventListener("reset", function () {
      window.setTimeout(function () {
        predictForm.querySelectorAll(".field-grid").forEach(function (grid) {
          grid.querySelectorAll('input[type="number"]').forEach(function (input) {
            input.value = input.defaultValue;
            const range = document.querySelector('[data-range-for="' + input.id + '"]');
            if (range) range.value = input.defaultValue;
            const counter = input.closest(".field")?.querySelector(".range-value");
            if (counter) counter.textContent = input.defaultValue;
          });
        });
      }, 0);
    });
  }

  /* ---------- Scroll ke kartu hasil ---------- */
  const resultCard = document.querySelector("[data-autoscroll]");
  if (resultCard) {
    window.setTimeout(function () {
      resultCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 140);
  }

  /* ---------- Filter riwayat ---------- */
  const filterForm = document.getElementById("filter-form");
  if (filterForm) {
    const select = filterForm.querySelector('select[name="status"]');
    if (select) {
      select.addEventListener("change", function () {
        filterForm.submit();
      });
    }
  }

  /* ---------- Hapus data + modal ---------- */
  const modal = document.getElementById("delete-modal");
  const deleteForm = document.getElementById("delete-form");
  const deleteName = document.getElementById("delete-name");
  let pendingId = null;

  function closeModal() {
    if (!modal) return;
    modal.hidden = true;
    document.body.style.overflow = "";
    pendingId = null;
  }

  if (modal && deleteForm) {
    modal.querySelectorAll("[data-close-modal]").forEach(function (el) {
      el.addEventListener("click", closeModal);
    });
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && !modal.hidden) closeModal();
    });

    document.querySelectorAll(".btn-delete").forEach(function (button) {
      button.addEventListener("click", function () {
        pendingId = button.dataset.id;
        if (deleteName) deleteName.textContent = button.dataset.name || "data ini";
        deleteForm.action = "/history/" + pendingId + "/delete";
        modal.hidden = false;
        document.body.style.overflow = "hidden";
      });
    });

    deleteForm.addEventListener("submit", function (event) {
      event.preventDefault();
      if (!pendingId) return;

      const id = pendingId;
      const row = document.querySelector('tr[data-log-id="' + id + '"]');
      const submitButton = deleteForm.querySelector("button[type=submit]");
      if (submitButton) submitButton.disabled = true;

      fetch("/api/logs/" + encodeURIComponent(id), { method: "DELETE" })
        .then(function (response) {
          if (!response.ok) throw new Error("Gagal menghapus data.");
          return response.json();
        })
        .then(function (data) {
          closeModal();
          if (row) {
            const isRisk = row.querySelector(".badge-risk") !== null;
            row.classList.add("is-removing");
            window.setTimeout(function () {
              row.remove();
              updateHistoryCounters(isRisk);
              checkEmptyTable();
            }, 280);
          }
          showToast("Data #" + id + " berhasil dihapus.", "success");
        })
        .catch(function () {
          deleteForm.submit();
        })
        .finally(function () {
          if (submitButton) submitButton.disabled = false;
        });
    });
  }

  function updateHistoryCounters(isRisk) {
    const total = document.getElementById("stat-total");
    const stable = document.getElementById("stat-stable");
    const risk = document.getElementById("stat-risk");
    const toolbarCount = document.querySelector(".toolbar-count strong");

    if (total) total.textContent = Math.max(0, parseInt(total.textContent, 10) - 1);
    if (isRisk) {
      if (risk) risk.textContent = Math.max(0, parseInt(risk.textContent, 10) - 1);
    } else if (stable) {
      stable.textContent = Math.max(0, parseInt(stable.textContent, 10) - 1);
    }
    if (toolbarCount) toolbarCount.textContent = Math.max(0, parseInt(toolbarCount.textContent, 10) - 1);
  }

  function checkEmptyTable() {
    const tbody = document.querySelector("#history-table tbody");
    if (tbody && tbody.children.length === 0) {
      window.location.href = "/history";
    }
  }

  /* ---------- Notifikasi dari query string ---------- */
  if (window.AQUASENSE_HISTORY && window.AQUASENSE_HISTORY.hasDeletedParam) {
    showToast("Data riwayat berhasil dihapus.", "success");
  }

})();
