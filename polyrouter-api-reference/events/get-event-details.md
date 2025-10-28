# Get event details

> Retrieve detailed information about a specific event using platform-native IDs

## OpenAPI

````yaml openapi.json get /events/{event_id}
paths:
  path: /events/{event_id}
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
        event_id:
          schema:
            - type: string
              required: true
              description: Platform-native event identifier
      query:
        include_raw:
          schema:
            - type: boolean
              required: false
              description: Include raw data from the platform API
        with_nested_markets:
          schema:
            - type: boolean
              required: false
              description: Include nested markets in the response
      header: {}
      cookie: {}
    body: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              events:
                allOf:
                  - type: array
                    items:
                      $ref: '#/components/schemas/EventV2'
              pagination:
                allOf:
                  - $ref: '#/components/schemas/Pagination'
              meta:
                allOf:
                  - $ref: '#/components/schemas/Meta'
        examples:
          example:
            value:
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
        description: Event details retrieved successfully
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

````