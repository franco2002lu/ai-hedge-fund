# Get series details

> Retrieve detailed information about a specific series using platform-native IDs

## OpenAPI

````yaml openapi.json get /series/{series_id}
paths:
  path: /series/{series_id}
  method: get
  servers:
    - url: https://api.polyrouter.io/functions/v1
      description: Production server
  request:
    security:
      - title: ApiKeyAuth
        parameters:
          query: {}
          header:
            X-API-Key:
              type: apiKey
              description: API key for authentication
          cookie: {}
    parameters:
      path:
        series_id:
          schema:
            - type: string
              required: true
              description: Platform-native series identifier
      query:
        include_raw:
          schema:
            - type: boolean
              required: false
              description: Include raw data from the platform API
      header: {}
      cookie: {}
    body: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              series:
                allOf:
                  - type: array
                    items:
                      $ref: '#/components/schemas/SeriesV2'
              pagination:
                allOf:
                  - $ref: '#/components/schemas/Pagination'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              series:
                - id: <string>
                  platform: polymarket
                  platform_id: <string>
                  title: <string>
                  series_slug: <string>
                  description: <string>
                  image_url: <string>
                  category: <string>
                  tags:
                    - <string>
                  metadata: {}
                  events:
                    - id: <string>
                      platform: polymarket
                      platform_id: <string>
                      series_id: <string>
                      title: <string>
                      event_slug: <string>
                      description: <string>
                      image_url: <string>
                      resolution_source_url: <string>
                      event_start_at: '2023-11-07T05:31:56Z'
                      event_end_at: '2023-11-07T05:31:56Z'
                      last_synced_at: '2023-11-07T05:31:56Z'
                      market_count: 123
                      total_volume: 123
                      raw_data: {}
                  markets:
                    - id: <string>
                      platform: polymarket
                      platform_id: <string>
                      event_id: <string>
                      event_name: <string>
                      event_slug: <string>
                      title: <string>
                      market_slug: <string>
                      description: <string>
                      subcategory: <string>
                      source_url: <string>
                      status: open
                      market_type: binary
                      category: <string>
                      tags:
                        - <string>
                      outcomes:
                        - id: <string>
                          name: <string>
                      current_prices: {}
                      volume_24h: 123
                      volume_7d: 123
                      volume_total: 123
                      liquidity: 123
                      liquidity_score: 0.5
                      open_interest: 123
                      unique_traders: 123
                      fee_rate: 123
                      trading_fee: 123
                      withdrawal_fee: 123
                      created_at: '2023-11-07T05:31:56Z'
                      trading_start_at: '2023-11-07T05:31:56Z'
                      trading_end_at: '2023-11-07T05:31:56Z'
                      resolution_date: '2023-11-07T05:31:56Z'
                      resolved_at: '2023-11-07T05:31:56Z'
                      resolution_criteria: <string>
                      resolution_source: <string>
                      price_24h_changes: {}
                      price_7d_changes: {}
                      last_trades: {}
                      metadata: {}
                      raw_data: {}
                      last_synced_at: '2023-11-07T05:31:56Z'
                  last_synced_at: '2023-11-07T05:31:56Z'
                  raw_data: {}
              pagination:
                total: 123
                limit: 123
                offset: 123
                has_more: true
                next_offset: 123
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Series details retrieved successfully
    '401':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
        description: Authentication error response
    '404':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Resource not found error response
    '500':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Internal server error response
    '503':
      application/json:
        schemaArray:
          - type: object
            properties:
              error:
                allOf:
                  - $ref: '#/components/schemas/Error'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
              error:
                code: VALIDATION_ERROR
                message: <string>
                details: {}
                timestamp: '2023-11-07T05:31:56Z'
              meta:
                request_time: 123
                cache_hit: true
                data_freshness: '2023-11-07T05:31:56Z'
        description: Platform unavailable
  deprecated: false
  type: path
components:
  schemas:
    CurrentPrices:
      type: object
      additionalProperties:
        $ref: '#/components/schemas/PriceDetail'
      description: Current prices for each outcome
    PriceDetail:
      type: object
      required:
        - price
      properties:
        price:
          type: number
          minimum: 0
          maximum: 1
          description: Current market price
        bid:
          type: number
          minimum: 0
          maximum: 1
          description: Best bid price
        ask:
          type: number
          minimum: 0
          maximum: 1
          description: Best ask price
    Pagination:
      type: object
      required:
        - total
        - limit
        - offset
        - has_more
        - next_offset
      properties:
        total:
          type: number
          description: Total number of results
        limit:
          type: number
          description: Number of results per page
        offset:
          type: number
          description: Current page offset
        has_more:
          type: boolean
          description: Whether more pages exist
        next_offset:
          type: number
          description: Offset for next page
    Meta:
      type: object
      required:
        - request_time
        - cache_hit
        - data_freshness
      properties:
        request_time:
          type: number
          description: Request processing time (ms)
        cache_hit:
          type: boolean
          description: Whether response was served from cache
        data_freshness:
          type: string
          format: date-time
          description: Data freshness timestamp
    Error:
      type: object
      required:
        - code
        - message
        - timestamp
      properties:
        code:
          type: string
          description: Error code
          example: VALIDATION_ERROR
        message:
          type: string
          description: Error message
        details:
          type: object
          description: Additional error details
        timestamp:
          type: string
          format: date-time
          description: Error timestamp
    EventV2:
      type: object
      required:
        - id
        - platform
        - platform_id
        - title
        - last_synced_at
      properties:
        id:
          type: string
          description: Platform-native event identifier
        platform:
          type: string
          enum:
            - polymarket
            - kalshi
            - limitless
            - manifold
          description: Source platform
        platform_id:
          type: string
          description: Original platform event ID (same as id)
        series_id:
          type: string
          description: Associated series identifier
        title:
          type: string
          description: Event title
        event_slug:
          type: string
          description: URL-friendly event identifier
        description:
          type: string
          description: Detailed event description
        image_url:
          type: string
          format: uri
          description: Event image URL
        resolution_source_url:
          type: string
          format: uri
          description: Resolution source URL
        event_start_at:
          type: string
          format: date-time
          description: Event start timestamp (ISO 8601)
        event_end_at:
          type: string
          format: date-time
          description: Event end timestamp (ISO 8601)
        last_synced_at:
          type: string
          format: date-time
          description: Last data sync timestamp (ISO 8601)
        market_count:
          type: number
          description: Number of markets in this event
        total_volume:
          type: number
          description: Total trading volume across all markets
        raw_data:
          type: object
          description: Complete raw platform response
    MarketV2:
      type: object
      required:
        - id
        - platform
        - platform_id
        - title
        - status
        - market_type
        - created_at
        - last_synced_at
      properties:
        id:
          type: string
          description: Platform-native market identifier
        platform:
          type: string
          enum:
            - polymarket
            - kalshi
            - limitless
            - manifold
          description: Source platform
        platform_id:
          type: string
          description: Original platform market ID (same as id)
        event_id:
          type: string
          description: Associated event identifier
        event_name:
          type: string
          nullable: true
          description: Event title (if available)
        event_slug:
          type: string
          nullable: true
          description: URL-friendly event identifier
        title:
          type: string
          description: Market question/title
        market_slug:
          type: string
          description: URL-friendly market identifier
        description:
          type: string
          description: Detailed market description
        subcategory:
          type: string
          nullable: true
          description: Market subcategory
        source_url:
          type: string
          format: uri
          description: Original platform URL
        status:
          type: string
          enum:
            - open
            - closed
            - resolved
            - paused
          description: Market status
        market_type:
          type: string
          enum:
            - binary
            - categorical
            - scalar
          description: Market type
        category:
          type: string
          nullable: true
          description: Market category
        tags:
          type: array
          items:
            type: string
          nullable: true
          description: Array of market tags
        outcomes:
          type: array
          items:
            $ref: '#/components/schemas/OutcomeV2'
          description: Array of market outcomes
        current_prices:
          $ref: '#/components/schemas/CurrentPrices'
        volume_24h:
          type: number
          nullable: true
          description: 24-hour trading volume
        volume_7d:
          type: number
          nullable: true
          description: 7-day trading volume
        volume_total:
          type: number
          description: Total trading volume
        liquidity:
          type: number
          description: Market liquidity
        liquidity_score:
          type: number
          minimum: 0
          maximum: 1
          description: Calculated liquidity score (0-1)
        open_interest:
          type: number
          nullable: true
          description: Current open interest
        unique_traders:
          type: number
          nullable: true
          description: Number of unique traders
        fee_rate:
          type: number
          nullable: true
          description: Trading fee rate
        trading_fee:
          type: number
          nullable: true
          description: Trading fee
        withdrawal_fee:
          type: number
          nullable: true
          description: Withdrawal fee
        created_at:
          type: string
          format: date-time
          description: Market creation timestamp (ISO 8601)
        trading_start_at:
          type: string
          format: date-time
          description: Trading start timestamp (ISO 8601)
        trading_end_at:
          type: string
          format: date-time
          description: Trading end timestamp (ISO 8601)
        resolution_date:
          type: string
          format: date-time
          description: Market resolution timestamp (ISO 8601)
        resolved_at:
          type: string
          format: date-time
          nullable: true
          description: Market resolution timestamp (ISO 8601)
        resolution_criteria:
          type: string
          description: Market resolution criteria
        resolution_source:
          type: string
          format: uri
          description: Resolution source URL
        price_24h_changes:
          type: object
          additionalProperties:
            type: number
          description: 24-hour price changes per outcome
        price_7d_changes:
          type: object
          additionalProperties:
            type: number
          description: 7-day price changes per outcome
        last_trades:
          type: object
          nullable: true
          description: Last trade information
        metadata:
          type: object
          description: Platform-specific metadata
        raw_data:
          type: object
          description: Complete raw platform response
        last_synced_at:
          type: string
          format: date-time
          description: Last data sync timestamp (ISO 8601)
    OutcomeV2:
      type: object
      required:
        - id
        - name
      properties:
        id:
          type: string
          description: Outcome identifier
        name:
          type: string
          description: Outcome display name
    SeriesV2:
      type: object
      required:
        - id
        - platform
        - platform_id
        - title
        - last_synced_at
      properties:
        id:
          type: string
          description: Platform-native series identifier
        platform:
          type: string
          enum:
            - polymarket
            - kalshi
            - limitless
            - manifold
          description: Source platform
        platform_id:
          type: string
          description: Original platform series ID (same as id)
        title:
          type: string
          description: Series title
        series_slug:
          type: string
          description: URL-friendly series identifier
        description:
          type: string
          description: Series description
        image_url:
          type: string
          format: uri
          description: Series image URL
        category:
          type: string
          description: Series category
        tags:
          type: array
          items:
            type: string
          description: Array of series tags
        metadata:
          type: object
          description: Platform-specific metadata
        events:
          type: array
          items:
            $ref: '#/components/schemas/EventV2'
          description: Array of nested events (for individual series lookup)
        markets:
          type: array
          items:
            $ref: '#/components/schemas/MarketV2'
          description: Array of nested markets (for individual series lookup)
        last_synced_at:
          type: string
          format: date-time
          description: Last data sync timestamp (ISO 8601)
        raw_data:
          type: object
          description: Complete raw platform response

````