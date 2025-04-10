
    document.addEventListener('DOMContentLoaded', function() {
        // Tab functionality
        const tabButtons = document.querySelectorAll('.tab-button');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                // Remove active class from all buttons and contents
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Add active class to clicked button and corresponding content
                button.classList.add('active');
                const tabId = button.getAttribute('data-tab');
                document.getElementById(tabId).classList.add('active');
            });
        });
        
        // Set the first tab as active by default
        if (tabButtons.length > 0) {
            tabButtons[0].click();
        }
        
        // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                
                const targetId = this.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                
                if (targetElement) {
                    window.scrollTo({
                        top: targetElement.offsetTop - 70,
                        behavior: 'smooth'
                    });
                }
            });
        });
    });
    
    // Function to filter models by precision
    function filterModelsByPrecision(precision) {
        const modelCards = document.querySelectorAll('.model-card');
        
        modelCards.forEach(card => {
            if (precision === 'all' || card.getAttribute('data-precision') === precision) {
                card.style.display = 'block';
            } else {
                card.style.display = 'none';
            }
        });
        
        // Update active button
        document.querySelectorAll('.precision-filter').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`.precision-filter[data-precision="${precision}"]`).classList.add('active');
    }
    