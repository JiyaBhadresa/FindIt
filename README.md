# FindIt AI 🔍
### AI-Powered Lost & Found Platform

FindIt AI is a full-stack AI-powered lost and found platform designed to make recovering lost belongings easier, smarter, and faster.

Instead of relying on manual searches and scattered social media posts, FindIt AI uses artificial intelligence to identify potential matches between lost and found items based on images, descriptions, attributes, location, and time.

The platform also provides a secure claim and verification workflow to help users establish ownership before an item is returned.

---

## ✨ Key Features

### 1. User Authentication
- User registration and login.
- JWT-based authentication.
- Protected routes and session management.

### 2. Lost & Found Reporting
- Report lost and found belongings.
- Add item titles, descriptions, categories, colors, brands, and locations.
- Upload item images.
- Store images using Cloudinary.

### 3. AI-Powered Matching Engine

FindIt AI combines multiple matching signals to identify potentially related items.

| Matching Signal | Technology / Approach |
|---|---|
| Text Similarity | Sentence Transformers |
| Image Similarity | CLIP (Vision Model) |
| Attribute Matching | Category, color, brand and features |
| Location Matching | Location and geographic proximity |
| Time Matching | Lost/found date comparison |

**Hybrid Matching:** Combines the individual signals into a weighted ranking score and dynamically adjusts weights when certain information is unavailable.

### 4. Smart Match Management
- View ranked potential matches.
- Explore detailed matching score breakdowns.
- Express interest in a potential match.
- Track match status and progress.

### 5. Claim & Ownership Verification
- Submit ownership claims.
- Answer confidential verification questions.
- Review claim decisions.
- Track claim history and recovery status.

### 6. Notifications
- In-app notifications.
- Unread notification tracking.
- Automatic notification polling.
- Mark individual or all notifications as read.

### 7. User Dashboard
- Browse lost and found reports.
- Search and filter items.
- View reported items and claim statistics.
- Manage account information.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Vite |
| Styling | Tailwind CSS |
| Backend | Python, FastAPI |
| Database | MongoDB Atlas |
| Database Driver | PyMongo Async API |
| Authentication | JWT |
| AI / NLP | Sentence Transformers |
| Computer Vision | CLIP |
| Similarity Calculation | Cosine Similarity |
| Image Storage | Cloudinary |
| API Communication | Axios |
| API Testing | FastAPI Swagger UI |

---

## 🏗️ System Architecture

```text
                 USER
                  |
                  v
          REACT FRONTEND
           (TypeScript)
                  |
                  v
          FASTAPI BACKEND
                  |
       +----------+----------+
       |          |          |
       v          v          v
   MongoDB    AI Matching  Cloudinary
   Atlas       Engine     Image Storage
                  |
       +----------+----------+
       |          |          |
       v          v          v
     Text       Image     Attributes
   Similarity  Similarity  Matching
       |          |          |
       +----------+----------+
                  |
                  v
         Location & Time
             Matching
                  |
                  v
          HYBRID SCORING
                  |
                  v
        RANKED POTENTIAL MATCHES
                  |
                  v
       CLAIM & VERIFICATION
                  |
                  v
            ITEM RECOVERY
```

---

## ⚙️ Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.11+
- Node.js and npm
- MongoDB Atlas account
- Cloudinary account

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/findit-ai.git
cd findit-ai
```

### 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -e .
```

Configure your environment variables in a `.env` file:

```env
MONGODB_URI=your_mongodb_connection_string
DB_NAME=findit_ai

CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
```

Add any additional environment variables required by your backend configuration, such as JWT settings.

Start the backend:

```bash
.venv/bin/uvicorn main:app --reload --port 8001
```

### 3. Frontend Setup

Open a second terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Configure the frontend API URL in its environment file using the variable name expected by your Axios configuration. For example:

```env
VITE_API_URL=http://127.0.0.1:8001
```

Start the frontend:

```bash
npm run dev
```

Frontend URL:


---

## 🧠 How AI Matching Works

When a user reports a lost or found item, FindIt AI evaluates potential matches using multiple signals.

1. **Text Matching:** Understands descriptions with different wording using semantic embeddings.
2. **Image Matching:** Uses CLIP embeddings to compare visual similarities, including objects photographed from different angles.
3. **Attribute Matching:** Compares item categories, colors, brands, and distinctive characteristics.
4. **Location Matching:** Evaluates geographical proximity and location information.
5. **Time Matching:** Considers the difference between lost and found dates.
6. **Hybrid Scoring:** Combines available signals into a final ranking score.

The system ranks potential matches to help users investigate. A similarity score is not a probability of ownership, and AI does not independently establish ownership.

---

## 🔐 Security & Privacy

- JWT-based authentication.
- Protected application routes.
- Confidential ownership verification information.
- Separate user-facing and verification information.
- Human review in the ownership verification process.
- Environment variables for sensitive configuration.

Never commit `.env` files, database credentials, API keys, or private user information.

---

## 🚀 Future Improvements

- Precomputed AI embeddings for faster matching.
- Optimized image-processing pipelines.
- Advanced duplicate-report detection.
- Admin dashboard and moderation tools.
- Improved fraud and suspicious-claim detection.
- Deployment with production monitoring.
- Analytics for successful item recoveries.

---

## 🎯 Project Goal

The goal of FindIt AI is to transform traditional lost-and-found systems into an intelligent, accessible, and privacy-conscious recovery platform.

By combining natural language processing, computer vision, and multi-factor matching, the project explores how AI can solve a practical everyday problem.

---

## 👩‍💻 Author

**Jiya Bhadresa**

Information Technology Student  
Sardar Vallabhbhai Patel Institute of Technology (SVIT)

---

*Built with Python, React, MongoDB, and AI.*
