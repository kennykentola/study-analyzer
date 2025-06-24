


document.addEventListener('DOMContentLoaded', () => {
    console.log('script.js loaded');
    const elements = {
        fileUpload: document.getElementById('file-upload'),
        analyzeButton: document.getElementById('analyze-file'),
        summarizeButton: document.getElementById('summarize-file'),
        summaryLength: document.getElementById('summary-length'),
        summaryType: document.getElementById('summary-type'),
        analysisResult: document.getElementById('analysis-result'),
        timePerPage: document.getElementById('time-per-page'),
        timeUnit: document.getElementById('time-unit'),
        savePrefsButton: document.getElementById('save-prefs'),
        chatbotQuestion: document.getElementById('chatbot-input'),
        chatbotAsk: document.getElementById('chatbot-ask'),
        chatbotConversation: document.getElementById('chatbot-conversation'),
        chatbotSectionSelect: document.getElementById('chatbot-section-select'),
        languageSelect: document.getElementById('language-select'),
        searchButton: document.getElementById('search-button'),
        searchInput: document.getElementById('search-input'),
        sectionSelect: document.getElementById('section-select'),
        searchResults: document.getElementById('search-results')
    };

    const missingElements = Object.entries(elements)
        .filter(([key, el]) => !el)
        .map(([key]) => key);
    if (missingElements.length) {
        console.error('Missing DOM elements:', missingElements);
        elements.analysisResult.innerHTML = `<p class="text-red-500">Error: Missing UI elements (${missingElements.join(', ')}). Check index.html.</p>`;
        return;
    }

    let currentLang = 'en';
    let currentFilename = null;
    let sections = [];
    let conversationHistory = [];

    async function loadTranslations(lang) {
        try {
            const response = await fetch(`/api/translations/${lang}`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const translations = await response.json();
            elements.chatbotQuestion.placeholder = translations.chatbot_placeholder || 'Ask a question...';
            elements.chatbotAsk.textContent = translations.chatbot_button || 'Ask';
            elements.analyzeButton.textContent = translations.analyze_button || 'Analyze File';
            elements.summarizeButton.textContent = translations.summarize_button || 'Summarize File';
            elements.savePrefsButton.textContent = translations.save_prefs || 'Save Preferences';
            elements.searchButton.textContent = translations.search_button || 'Search';
            currentLang = lang;
            console.log(`Translations loaded for ${lang}`);
        } catch (error) {
            console.error('Error loading translations:', error);
            elements.analysisResult.innerHTML = `<p class="text-red-500">Failed to load translations: ${error.message}</p>`;
        }
    }

    async function loadPreferences() {
        try {
            const response = await fetch('/api/preferences');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const prefs = await response.json();
            elements.timePerPage.value = prefs.time_per_page || 45;
            elements.timeUnit.value = prefs.time_unit || 'minutes';
            console.log('Preferences loaded:', prefs);
        } catch (error) {
            console.error('Error loading preferences:', error);
            elements.analysisResult.innerHTML = `<p class="text-red-500">Failed to load preferences: ${error.message}</p>`;
        }
    }

    elements.savePrefsButton.addEventListener('click', async () => {
        try {
            const timePerPage = parseFloat(elements.timePerPage.value);
            const timeUnit = elements.timeUnit.value;
            if (isNaN(timePerPage) || timePerPage <= 0) throw new Error('Time per page must be a positive number');
            let secondsPerPage = timePerPage;
            let maxInput;
            switch (timeUnit) {
                case 'seconds':
                    maxInput = 3600;
                    break;
                case 'minutes':
                    maxInput = 60;
                    secondsPerPage *= 60;
                    break;
                case 'hours':
                    maxInput = 1;
                    secondsPerPage *= 3600;
                    break;
                default:
                    throw new Error('Invalid time unit');
            }
            if (timePerPage > maxInput) throw new Error(`Time per page cannot exceed ${maxInput} ${timeUnit}`);
            const prefs = { time_per_page: timePerPage, time_unit: timeUnit };
            const response = await fetch('/api/preferences', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(prefs)
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}: ${(await response.json()).error}`);
            elements.analysisResult.innerHTML = `<p class="text-green-500">Preferences saved.</p>`;
        } catch (error) {
            console.error('Error saving preferences:', error);
            elements.analysisResult.innerHTML = `<p class="text-red-500">Failed to save preferences: ${error.message}</p>`;
        }
    });

    function updateSections(newSections) {
        sections = newSections || [];
        elements.chatbotSectionSelect.innerHTML = '<option value="">All Content</option>';
        elements.sectionSelect.innerHTML = '<option value="">All Sections</option>';
        sections.forEach(section => {
            const option = document.createElement('option');
            option.value = section.title;
            option.textContent = section.title;
            elements.chatbotSectionSelect.appendChild(option.cloneNode(true));
            elements.sectionSelect.appendChild(option);
        });
    }

    function updateConversation() {
        console.log('Updating conversation with history:', conversationHistory);
        elements.chatbotConversation.innerHTML = '';
        conversationHistory.forEach((msg, index) => {
            const isUser = msg.type === 'user';
            const messageDiv = document.createElement('div');
            messageDiv.className = `p-2 mb-2 rounded-lg ${isUser ? 'bg-blue-100 text-right' : 'bg-gray-200 text-left'}`;
            messageDiv.innerHTML = `
                <p class="text-sm"><strong>${isUser ? 'You' : 'Bot'}:</strong> ${msg.text}</p>
                ${msg.sources && msg.sources.length ? `<p class="text-xs text-gray-600">Sources: ${msg.sources.join(', ')}</p>` : ''}
                ${msg.needsAnswer ? `
                    <div class="mt-2">
                        <input type="text" id="correct-answer-${index}" class="border p-1 rounded w-full text-sm" placeholder="Enter correct answer...">
                        <button id="submit-answer-${index}" class="bg-blue-500 text-white p-1 rounded mt-1 text-xs hover:bg-blue-600">Submit</button>
                    </div>
                ` : ''}
            `;
            elements.chatbotConversation.appendChild(messageDiv);
            if (msg.needsAnswer) {
                document.getElementById(`submit-answer-${index}`).addEventListener('click', () => submitAnswer(index));
            }
        });
        elements.chatbotConversation.scrollTop = elements.chatbotConversation.scrollHeight;
    }

    async function submitAnswer(index) {
        const correctAnswerInput = document.getElementById(`correct-answer-${index}`);
        const correctAnswer = correctAnswerInput.value.trim();
        if (!correctAnswer) {
            alert('Please provide an answer');
            return;
        }
        try {
            const question = conversationHistory[index].question;
            console.log(`Submitting answer for question: ${question}`);
            const response = await fetch('/api/learn', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, answer: correctAnswer })
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
            conversationHistory[index].needsAnswer = false;
            conversationHistory[index].text = `Learned answer: ${correctAnswer}`;
            conversationHistory.push({
                type: 'bot',
                text: `The answer has been saved. Ask "${question}" again to verify.`,
                sources: [],
                needsAnswer: false
            });
            updateConversation();
            console.log(`Learned answer for question: ${question}`);
        } catch (error) {
            console.error('Error learning answer:', error);
            elements.chatbotConversation.innerHTML += `<p class="text-red-500 text-sm">Failed to learn answer: ${error.message}</p>`;
            elements.chatbotConversation.scrollTop = elements.chatbotConversation.scrollHeight;
        }
    }

    elements.analyzeButton.addEventListener('click', async () => {
        console.log('Analyze button clicked');
        elements.analysisResult.innerHTML = '<div class="loader"></div>';
        await new Promise(resolve => setTimeout(resolve, 500));
        const file = elements.fileUpload.files[0];
        if (!file) {
            elements.analysisResult.innerHTML = `<p class="text-red-500">No file selected</p>`;
            return;
        }
        const formData = new FormData();
        formData.append('file', file);
        try {
            const response = await fetch('/api/files/analyze', { method: 'POST', body: formData });
            if (!response.ok) throw new Error((await response.json()).error);
            const result = await response.json();
            currentFilename = result.filename;
            elements.analysisResult.innerHTML = `
                <p>File: ${result.filename}</p>
                <p>Pages: ${result.page_count}</p>
                <p>Estimated Study Time: ${result.formatted_time}</p>
            `;
            updateSections([]);
            conversationHistory = [];
            updateConversation();
        } catch (error) {
            console.error('Error analyzing file:', error);
            elements.analysisResult.innerHTML = `<p class="text-red-500">Error analyzing file: ${error.message}</p>`;
        }
    });

    elements.summarizeButton.addEventListener('click', async () => {
        console.log('Summarize button clicked');
        elements.analysisResult.innerHTML = '<div class="loader"></div>';
        await new Promise(resolve => setTimeout(resolve, 500));
        const file = elements.fileUpload.files[0];
        if (!file) {
            elements.analysisResult.innerHTML = `<p class="text-red-500">No file selected</p>`;
            return;
        }
        const formData = new FormData();
        formData.append('file', file);
        formData.append('num_sentences', elements.summaryLength.value);
        formData.append('summary_type', elements.summaryType.value);
        try {
            const response = await fetch('/api/files/summarize', { method: 'POST', body: formData });
            if (!response.ok) throw new Error((await response.json()).error);
            const result = await response.json();
            currentFilename = result.filename;
            let resultHTML = `
                <p>File: ${result.filename}</p>
                <p>Pages: ${result.page_count}</p>
                <p>Estimated Study Time: ${result.formatted_time}</p>
            `;
            if (result.sections && result.sections.length) {
                resultHTML += `<h3 class="font-semibold mt-4">Summary:</h3>`;
                result.sections.forEach(section => {
                    resultHTML += `
                        <div class="mt-2">
                            <p class="font-medium">${section.title} (Pages ${section.start_page}-${section.end_page})</p>
                            <p>${section.summary || 'No summary available'}</p>
                        </div>
                    `;
                });
                updateSections(result.sections);
            } else {
                resultHTML += `<h3 class="font-semibold mt-4">Summary:</h3><p>${result.summary || 'No summary available'}</p>`;
                updateSections([]);
            }
            elements.analysisResult.innerHTML = resultHTML;
            conversationHistory = [];
            updateConversation();
        } catch (error) {
            console.error('Error summarizing file:', error);
            elements.analysisResult.innerHTML = `<p class="text-red-500">Error summarizing file: ${error.message}</p>`;
        }
    });

    elements.searchButton.addEventListener('click', async () => {
        console.log('Search button clicked');
        const query = elements.searchInput.value.trim();
        const sectionTitle = elements.sectionSelect.value;
        if (!query) {
            elements.searchResults.innerHTML = `<p class="text-red-500">No search term provided</p>`;
            return;
        }
        if (!currentFilename) {
            elements.searchResults.innerHTML = `<p class="text-red-500">No file uploaded</p>`;
            return;
        }
        elements.searchResults.innerHTML = '<div class="loader"></div>';
        await new Promise(resolve => setTimeout(resolve, 500));
        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: query,
                    filename: currentFilename,
                    section_title: sectionTitle
                })
            });
            if (!response.ok) throw new Error((await response.json()).error);
            const result = await response.json();
            elements.searchResults.innerHTML = `
                <p><strong>Result:</strong> ${result.answer}</p>
                ${result.sources && result.sources.length ? `<p class="text-xs text-gray-600">Sources: ${result.sources.join(', ')}</p>` : ''}
            `;
        } catch (error) {
            console.error('Error searching document:', error);
            elements.searchResults.innerHTML = `<p class="text-red-500">Error searching document: ${error.message}</p>`;
        }
    });

    elements.chatbotAsk.addEventListener('click', async () => {
        console.log('Chatbot ask button clicked');
        const question = elements.chatbotQuestion.value.trim();
        if (!question) {
            elements.chatbotConversation.innerHTML += `<p class="text-red-500 text-sm">No question provided</p>`;
            elements.chatbotConversation.scrollTop = elements.chatbotConversation.scrollHeight;
            return;
        }
        conversationHistory.push({ type: 'user', text: question, question });
        updateConversation();
        elements.chatbotQuestion.value = '';
        elements.chatbotConversation.innerHTML += '<div class="loader"></div>';
        await new Promise(resolve => setTimeout(resolve, 500));
        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question,
                    filename: currentFilename,
                    section_title: elements.chatbotSectionSelect.value
                })
            });
            elements.chatbotConversation.innerHTML = ''; // Remove loader
            if (!response.ok) throw new Error((await response.json()).error);
            const result = await response.json();
            conversationHistory.push({
                type: 'bot',
                text: result.answer || 'No answer provided',
                sources: result.sources || [],
                needsAnswer: result.answer.includes('Please provide the correct answer'),
                question
            });
            updateConversation();
        } catch (error) {
            console.error('Error asking question:', error);
            elements.chatbotConversation.innerHTML = ''; // Remove loader
            conversationHistory.push({ type: 'bot', text: `Error asking question: ${error.message}` });
            updateConversation();
        }
    });

    elements.languageSelect.addEventListener('change', () => {
        loadTranslations(elements.languageSelect.value);
    });

    elements.timePerPage.addEventListener('input', () => {
        const unit = elements.timeUnit.value;
        let max;
        if (unit === 'seconds') max = 3600;
        else if (unit === 'minutes') max = 60;
        else if (unit === 'hours') max = 1;
        elements.timePerPage.max = max;
        if (parseFloat(elements.timePerPage.value) > max) {
            elements.timePerPage.value = max;
        }
    });

    elements.timeUnit.addEventListener('change', () => {
        elements.timePerPage.dispatchEvent(new Event('input'));
    });

    loadTranslations('en');
    loadPreferences();
});