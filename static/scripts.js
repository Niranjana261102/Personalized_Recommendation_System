<script>
// Form Validation and Submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('recommendationForm');

    if (form) {
        form.addEventListener('submit', function(e) {
            const submitBtn = form.querySelector('button[type="submit"]');
            const originalText = submitBtn.innerHTML;

            // Add loading state
            submitBtn.innerHTML = '<span class="loading me-2"></span>Getting Recommendations...';
            submitBtn.disabled = true;

            // Basic validation
            const userId = document.getElementById('user_id').value;
            const algorithm = document.getElementById('algorithm').value;

            if (!userId || !algorithm) {
                e.preventDefault();
                alert('Please fill in all required fields.');
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
                return;
            }

            if (userId < 1) {
                e.preventDefault();
                alert('User ID must be a positive number.');
                submitBtn.innerHTML = originalText;
                submitBtn.disabled = false;
                return;
            }
        });
    }

    // Initialize charts if on analytics page
    if (document.getElementById('topMoviesChart')) {
        initializeCharts();
    }

    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-info)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});

// Chart.js initialization for analytics
function initializeCharts() {
    // Top Movies Chart
    const topMoviesCtx = document.getElementById('topMoviesChart');
    if (topMoviesCtx) {
        new Chart(topMoviesCtx, {
            type: 'bar',
            data: {
                labels: ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4', 'Movie 5'],
                datasets: [{
                    label: 'Average Rating',
                    data: [4.8, 4.7, 4.6, 4.5, 4.4],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(255, 205, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 99, 132, 1)',
                        'rgba(255, 205, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)'
                    ],
                    borderWidth: 2,
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 5
                    }
                }
            }
        });
    }

    // Genre Distribution Chart
    const genreCtx = document.getElementById('genreChart');
    if (genreCtx) {
        new Chart(genreCtx, {
            type: 'doughnut',
            data: {
                labels: ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi'],
                datasets: [{
                    data: [25, 20, 18, 12, 15, 10],
                    backgroundColor: [
                        '#FF6384',
                        '#36A2EB',
                        '#FFCE56',
                        '#4BC0C0',
                        '#9966FF',
                        '#FF9F40'
                    ],
                    borderWidth: 3,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    // Rating Distribution Chart
    const ratingDistCtx = document.getElementById('ratingDistChart');
    if (ratingDistCtx) {
        new Chart(ratingDistCtx, {
            type: 'line',
            data: {
                labels: ['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0'],
                datasets: [{
                    label: 'Number of Ratings',
                    data: [120, 250, 450, 780, 1200, 1500, 1800, 1200, 800],
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.4,
                    fill: true,
                    borderWidth: 3,
                    pointBackgroundColor: 'rgba(75, 192, 192, 1)',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

// Utility Functions
function showToast(message, type = 'info') {
    const toastContainer = document.getElementById('toast-container') || createToastContainer();
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">${message}</div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;

    toastContainer.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();

    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.style.zIndex = '1055';
    document.body.appendChild(container);
    return container;
}

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add Chart.js CDN if not already loaded
if (typeof Chart === 'undefined') {
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js';
    script.onload = function() {
        if (document.getElementById('topMoviesChart')) {
            initializeCharts();
        }
    };
    document.head.appendChild(script);
}
</script>
