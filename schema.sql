-- Booked meeting appointments table (MySQL)
-- Run this against your database to create the table.

CREATE TABLE IF NOT EXISTS meeting_appointments (
    id                  BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    first_name          VARCHAR(100)   NOT NULL COMMENT 'Client first name',
    last_name           VARCHAR(100)   NOT NULL COMMENT 'Client last name',
    email               VARCHAR(255)   NOT NULL COMMENT 'Client email',
    start_date          DATE           NOT NULL COMMENT 'Appointment start date',
    start_time          TIME           NOT NULL COMMENT 'Appointment start time',
    end_date            DATE           NOT NULL COMMENT 'Appointment end date',
    end_time            TIME           NOT NULL COMMENT 'Appointment end time',
    service_requested   VARCHAR(255)   NOT NULL COMMENT 'Service the client requested',
    freelancer_id       BIGINT UNSIGNED NULL     COMMENT 'ID of the freelancer (links to freelancers table if present)',
    client_id           BIGINT UNSIGNED NULL     COMMENT 'ID of the client (links to clients table if present)',
    created_at          DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_freelancer_id (freelancer_id),
    INDEX idx_client_id (client_id),
    INDEX idx_start_date (start_date),
    INDEX idx_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
